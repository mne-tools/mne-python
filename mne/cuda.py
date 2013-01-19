# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.fftpack import fft, ifft
try:
    import pycuda.gpuarray as gpuarray
    from pycuda.driver import mem_get_info
    from scikits.cuda import fft as cudafft
except ImportError:
    pass

import logging
logger = logging.getLogger('mne')

from .utils import sizeof_fmt


# Support CUDA for FFTs; requires scikits.cuda and pycuda
cuda_capable = False
cuda_multiply_inplace_complex64 = None
cuda_multiply_inplace_complex32 = None
requires_cuda = np.testing.dec.skipif(True, 'CUDA not initialized')


def init_cuda():
    """Initialize CUDA functionality

    This function attempts to load the necessary interfaces
    (hardware connectivity) to run CUDA-based filering. This
    function should only need to be run once per session.

    If the config var (set via mne.utils.set_config or in ENV)
    MNE_USE_CUDA == 'true', this function will be executed when
    importing mne. If this variable is not set, this function can
    be manually executed.
    """
    global cuda_capable
    global cuda_multiply_inplace_complex64
    global cuda_multiply_inplace_complex32
    global requires_cuda
    if cuda_capable is True:
        logger.info('CUDA previously enabled, currently %s available memory'
                    % sizeof_fmt(mem_get_info()[0]))
        return
    # Triage possible errors for informative messaging
    cuda_capable = False
    try:
        import pycuda.gpuarray
        import pycuda.driver
    except ImportError:
        logger.warn('module pycuda not found, CUDA not enabled')
    else:
        try:
            # Initialize CUDA; happens with importing autoinit
            import pycuda.autoinit
        except ImportError:
            logger.warn('pycuda.autoinit could not be imported, likely '
                        'a hardware error, CUDA not enabled')
        else:
            # Make our multiply inplace kernel
            try:
                from pycuda.elementwise import ElementwiseKernel
                # let's construct our own CUDA multiply in-place function
                dtype = 'pycuda::complex<double>'
                cuda_multiply_inplace_complex64 = \
                    ElementwiseKernel(dtype + ' *a, ' + dtype + ' *b',
                                      'b[i] = a[i] * b[i]', 'multiply_inplace')
            except:
                # This should never happen
                raise RuntimeError('pycuda ElementwiseKernel could not be '
                                   'constructed, please report this issue '
                                   'to mne-python developers with your '
                                   'system information and pycuda version')
            else:
                # Make sure scikits.cuda is installed
                try:
                    from scikits.cuda import fft as cudafft
                except ImportError:
                    logger.warn('modudle scikits.cuda not found, CUDA not '
                                'enabled')
                else:
                    # Make sure we can use 64-bit FFTs
                    try:
                        fft_plan = cudafft.Plan(16, np.float64, np.complex128)
                        del fft_plan
                    except:
                        logger.warn('Device does not support 64-bit FFTs, '
                                    'CUDA not enabled')
                    else:
                        cuda_capable = True
                        # Figure out limit for CUDA FFT calculations
                        logger.info('Enabling CUDA with %s available memory'
                                    % sizeof_fmt(mem_get_info()[0]))
    requires_cuda = np.testing.dec.skipif(not cuda_capable,
                                          'CUDA not initialized')


###############################################################################
# Repeated FFT multiplication

def setup_cuda_fft_multiply_repeated(n_jobs, h_fft):
    """Set up repeated CUDA FFT multiplication with a given filter

    Parameters
    ----------
    n_jobs : int | str
        If n_jobs == 'cuda', the function will attempt to set up for CUDA
        FFT multiplication.
    h_fft : array
        The filtering function that will be used repeatedly.
        If n_jobs='cuda', this function will be shortened (since CUDA
        assumes FFTs of real signals are half the length of the signal)
        and turned into a gpuarray.

    Returns
    -------
    n_jobs : int
        Sets n_jobs = 1 if n_jobs == 'cuda' was passed in, otherwise
        original n_jobs is passed.
    cuda_dict : dict
        Dictionary with the following CUDA-related variables:
            use_cuda : bool
                Whether CUDA should be used.
            fft_plan : instance of FFTPlan
                FFT plan to use in calculating the FFT.
            ifft_plan : instance of FFTPlan
                FFT plan to use in calculating the IFFT.
            x_fft : instance of gpuarray
                Empty allocated GPU space for storing the result of the
                frequency-domain multiplication.
            x : instance of gpuarray
                Empty allocated GPU space for the data to filter.
    h_fft : array | instance of gpuarray
        This will either be a gpuarray (if CUDA enabled) or np.ndarray.
        If CUDA is enabled, h_fft will be modified appropriately for use
        with filter.fft_multiply().

    Notes
    -----
    This function is designed to be used with fft_multiply_repeated().
    """
    cuda_dict = dict(use_cuda=False, fft_plan=None, ifft_plan=None,
                     x_fft=None, x=None, fft_len=None)
    n_fft = len(h_fft)
    if n_jobs == 'cuda':
        n_jobs = 1
        if cuda_capable:
            # set up all arrays necessary for CUDA
            cuda_fft_len = int((n_fft + (n_fft % 2)) / 2 + 1)
            use_cuda = False
            # try setting up for float64
            try:
                fft_plan = cudafft.Plan(n_fft, np.float64, np.complex128)
                ifft_plan = cudafft.Plan(n_fft, np.complex128, np.float64)
                x_fft = gpuarray.empty(cuda_fft_len, np.complex128)
                x = gpuarray.empty(int(n_fft), np.float64)
                cuda_h_fft = h_fft[:cuda_fft_len].astype('complex128')
                # do the IFFT normalization now so we don't have to later
                cuda_h_fft /= len(h_fft)
                h_fft = gpuarray.to_gpu(cuda_h_fft)
                dtype = np.float64
                multiply_inplace = cuda_multiply_inplace_complex64
            except:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
            else:
                use_cuda = True

            if use_cuda is True:
                logger.info('Using CUDA for FFT FIR filtering')
                cuda_dict['use_cuda'] = True
                cuda_dict['fft_plan'] = fft_plan
                cuda_dict['ifft_plan'] = ifft_plan
                cuda_dict['x_fft'] = x_fft
                cuda_dict['x'] = x
                cuda_dict['dtype'] = dtype
                cuda_dict['multiply_inplace'] = multiply_inplace
        else:
            logger.info('CUDA not used, CUDA has not been initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict, h_fft


def fft_multiply_repeated(h_fft, x, cuda_dict=dict(use_cuda=False)):
    """Do FFT multiplication by a filter function (possibly using CUDA)

    Parameters
    ----------
    h_fft : 1-d array or gpuarray
        The filtering array to apply.
    x : 1-d array
        The array to filter.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    if not cuda_dict['use_cuda']:
        # do the fourier-domain operations
        x = np.real(ifft(h_fft * fft(x), overwrite_x=True)).ravel()
    else:
        # do the fourier-domain operations, results in second param
        cuda_dict['x'].set(x.astype(cuda_dict['dtype']))
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        cuda_dict['multiply_inplace'](h_fft, cuda_dict['x_fft'])
        # If we wanted to do it locally instead of using our own kernel:
        # cuda_seg_fft.set(cuda_seg_fft.get() * h_fft)
        cudafft.ifft(cuda_dict['x_fft'], cuda_dict['x'],
                     cuda_dict['ifft_plan'], False)
        x = cuda_dict['x'].get().astype(x.dtype, subok=True, copy=False)
    return x


###############################################################################
# FFT Resampling

def setup_cuda_fft_resample(n_jobs, W, new_len):
    """Set up CUDA FFT resampling

    Parameters
    ----------
    n_jobs : int | str
        If n_jobs == 'cuda', the function will attempt to set up for CUDA
        FFT resampling.
    W : array
        The filtering function to be used during resampling.
        If n_jobs='cuda', this function will be shortened (since CUDA
        assumes FFTs of real signals are half the length of the signal)
        and turned into a gpuarray.
    new_len : int
        The size of the array following resampling.

    Returns
    -------
    n_jobs : int
        Sets n_jobs = 1 if n_jobs == 'cuda' was passed in, otherwise
        original n_jobs is passed.
    cuda_dict : dict
        Dictionary with the following CUDA-related variables:
            use_cuda : bool
                Whether CUDA should be used.
            fft_plan : instance of FFTPlan
                FFT plan to use in calculating the FFT.
            ifft_plan : instance of FFTPlan
                FFT plan to use in calculating the IFFT.
            x_fft : instance of gpuarray
                Empty allocated GPU space for storing the result of the
                frequency-domain multiplication.
            y_fft : instance of gpuarray
                Empty allocated GPU space for storing the result of the
                resampling.
            x : instance of gpuarray
                Empty allocated GPU space for the data to resample.
            y : instance of gpuarray
                Empty allocated GPU space for the resampled data.
    W : array | instance of gpuarray
        This will either be a gpuarray (if CUDA enabled) or np.ndarray.
        If CUDA is enabled, W will be modified appropriately for use
        with filter.fft_multiply().

    Notes
    -----
    This function is designed to be used with fft_resample().
    """
    cuda_dict = dict(use_cuda=False, fft_plan=None, ifft_plan=None,
                     x_fft=None, x=None, y_fft=None, y=None)
    if n_jobs == 'cuda':
        n_jobs = 1
        if cuda_capable:
            use_cuda = False
            # try setting up for float64
            try:
                n_fft_x = len(W)
                cuda_fft_len_x = int((n_fft_x + (n_fft_x % 2)) / 2 + 1)
                n_fft_y = new_len
                cuda_fft_len_y = int((n_fft_y + (n_fft_y % 2)) / 2 + 1)
                fft_plan = cudafft.Plan(n_fft_x, np.float64, np.complex128)
                ifft_plan = cudafft.Plan(n_fft_y, np.complex128, np.float64)
                x_fft = gpuarray.empty(cuda_fft_len_x, np.complex128)
                y_fft = gpuarray.empty(cuda_fft_len_y, np.complex128)
                x = gpuarray.empty(int(n_fft_x), np.float64)
                y = gpuarray.empty(int(n_fft_y), np.float64)
                cuda_W = W[:cuda_fft_len_x].astype('complex128')
                # do the IFFT normalization now so we don't have to later
                cuda_W /= n_fft_y
                W = gpuarray.to_gpu(cuda_W)
                dtype = np.float64
                multiply_inplace = cuda_multiply_inplace_complex64
            except:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
            else:
                use_cuda = True

            if use_cuda is True:
                logger.info('Using CUDA for FFT FIR filtering')
                cuda_dict['use_cuda'] = True
                cuda_dict['fft_plan'] = fft_plan
                cuda_dict['ifft_plan'] = ifft_plan
                cuda_dict['x_fft'] = x_fft
                cuda_dict['y_fft'] = y_fft
                cuda_dict['x'] = x
                cuda_dict['y'] = y
                cuda_dict['dtype'] = dtype
                cuda_dict['multiply_inplace'] = multiply_inplace
        else:
            logger.info('CUDA not used, CUDA has not been initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict, W


def fft_resample(x, W, new_len, npad, to_remove,
                 cuda_dict=dict(use_cuda=False)):
    """Do FFT resampling with a filter function (possibly using CUDA)

    Parameters
    ----------
    x : 1-d array
        The array to resample.
    W : 1-d array or gpuarray
        The filtering function to apply.
    new_len : int
        The size of the output array (before removing padding).
    npad : int
        Amount of padding to apply before resampling.
    to_remove : int
        Number of samples to remove after resampling.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # add some padding at beginning and end to make this work a little cleaner
    x = _smart_pad(x, npad)
    N = int(np.minimum(new_len, len(x)))
    sl_1 = slice((N + 1) / 2)
    y_fft = np.zeros(new_len, np.complex128)
    if not cuda_dict['use_cuda']:
        x_fft = fft(x).ravel()
        x_fft *= W
        y_fft[sl_1] = x_fft[sl_1]
        sl_2 = slice(-(N - 1) / 2, None)
        y_fft[sl_2] = x_fft[sl_2]
        y = np.real(ifft(y_fft, overwrite_x=True)).ravel()
    else:
        # do the fourier-domain operations, results in second param
        cuda_dict['x'].set(x.astype(cuda_dict['dtype']))
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        cuda_dict['multiply_inplace'](W, cuda_dict['x_fft'])
        x_fft = cuda_dict['x_fft'].get()
        y_fft[sl_1] = x_fft[sl_1]
        cuda_dict['y_fft'].set(y_fft[:cuda_dict['y_fft'].size])
        cudafft.ifft(cuda_dict['y_fft'], cuda_dict['y'],
                     cuda_dict['ifft_plan'], False)
        y = cuda_dict['y'].get().astype(x.dtype, subok=True, copy=False)

    # now let's trim it back to the correct size (if there was padding)
    if to_remove > 0:
        keep = np.ones((new_len), dtype='bool')
        keep[:to_remove] = False
        keep[-to_remove:] = False
        y = np.compress(keep, y)

    return y


###############################################################################
# Misc

# this has to go in mne.cuda instead of mne.filter to avoid import errors
def _smart_pad(x, n_pad):
    """Pad vector x
    """
    return np.r_[2 * x[0] - x[n_pad:0:-1], x, 2 * x[-1] - x[-2:-n_pad - 2:-1]]
