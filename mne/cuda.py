# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.fftpack import fft, ifft, rfft, irfft

from .utils import sizeof_fmt, logger, get_config, warn, _explain_exception


# Support CUDA for FFTs; requires scikits.cuda and pycuda
_cuda_capable = False
_multiply_inplace_c128 = _halve_c128 = _real_c128 = None


def _get_cudafft():
    """Deal with scikit-cuda namespace change."""
    try:
        from skcuda import fft
    except ImportError:
        try:
            from scikits.cuda import fft
        except ImportError:
            fft = None
    return fft


def get_cuda_memory():
    """Get the amount of free memory for CUDA operations.

    Returns
    -------
    memory : str
        The amount of available memory as a human-readable string.
    """
    if not _cuda_capable:
        warn('CUDA not enabled, returning zero for memory')
        mem = 0
    else:
        from pycuda.driver import mem_get_info
        mem = mem_get_info()[0]
    return sizeof_fmt(mem)


def init_cuda(ignore_config=False):
    """Initialize CUDA functionality.

    This function attempts to load the necessary interfaces
    (hardware connectivity) to run CUDA-based filtering. This
    function should only need to be run once per session.

    If the config var (set via mne.set_config or in ENV)
    MNE_USE_CUDA == 'true', this function will be executed when
    the first CUDA setup is performed. If this variable is not
    set, this function can be manually executed.
    """
    global _cuda_capable, _multiply_inplace_c128, _halve_c128, _real_c128
    if _cuda_capable:
        return
    if not ignore_config and (get_config('MNE_USE_CUDA', 'false').lower() !=
                              'true'):
        logger.info('CUDA not enabled in config, skipping initialization')
        return
    # Triage possible errors for informative messaging
    _cuda_capable = False
    try:
        from pycuda import gpuarray, driver  # noqa: F401
        from pycuda.elementwise import ElementwiseKernel
    except ImportError:
        warn('module pycuda not found, CUDA not enabled')
        return
    try:
        # Initialize CUDA; happens with importing autoinit
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        warn('pycuda.autoinit could not be imported, likely a hardware error, '
             'CUDA not enabled%s' % _explain_exception())
        return
    # Make sure scikit-cuda is installed
    cudafft = _get_cudafft()
    if cudafft is None:
        warn('module scikit-cuda not found, CUDA not enabled')
        return

    # let's construct our own CUDA multiply in-place function
    _multiply_inplace_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a, pycuda::complex<double> *b',
        'b[i] *= a[i]', 'multiply_inplace')
    _halve_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a', 'a[i] /= 2.0', 'halve_value')
    _real_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a', 'a[i] = real(a[i])', 'real_value')

    # Make sure we can use 64-bit FFTs
    try:
        cudafft.Plan(16, np.float64, np.complex128)  # will get auto-GC'ed
    except Exception:
        warn('Device does not appear to support 64-bit FFTs, CUDA not '
             'enabled%s' % _explain_exception())
        return
    _cuda_capable = True
    # Figure out limit for CUDA FFT calculations
    logger.info('Enabling CUDA with %s available memory' % get_cuda_memory())


###############################################################################
# Repeated FFT multiplication

def setup_cuda_fft_multiply_repeated(n_jobs, h_fft):
    """Set up repeated CUDA FFT multiplication with a given filter.

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
                     x_fft=None, x=None)
    n_fft = len(h_fft)
    cuda_fft_len = int((n_fft - (n_fft % 2)) / 2 + 1)
    if n_jobs == 'cuda':
        n_jobs = 1
        init_cuda()
        if _cuda_capable:
            from pycuda import gpuarray
            cudafft = _get_cudafft()
            # set up all arrays necessary for CUDA
            # try setting up for float64
            try:
                # do the IFFT normalization now so we don't have to later
                h_fft = gpuarray.to_gpu(h_fft[:cuda_fft_len]
                                        .astype('complex_') / len(h_fft))
                cuda_dict.update(
                    use_cuda=True,
                    fft_plan=cudafft.Plan(n_fft, np.float64, np.complex128),
                    ifft_plan=cudafft.Plan(n_fft, np.complex128, np.float64),
                    x_fft=gpuarray.empty(cuda_fft_len, np.complex128),
                    x=gpuarray.empty(int(n_fft), np.float64))
                logger.info('Using CUDA for FFT FIR filtering')
            except Exception as exp:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large: "%s"), falling back to '
                            'n_jobs=1' % str(exp))
        else:
            logger.info('CUDA not used, CUDA could not be initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict, h_fft


def fft_multiply_repeated(h_fft, x, cuda_dict=dict(use_cuda=False)):
    """Do FFT multiplication by a filter function (possibly using CUDA).

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
        cudafft = _get_cudafft()
        # do the fourier-domain operations, results in second param
        cuda_dict['x'].set(x.astype(np.float64))
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        _multiply_inplace_c128(h_fft, cuda_dict['x_fft'])
        # If we wanted to do it locally instead of using our own kernel:
        # cuda_seg_fft.set(cuda_seg_fft.get() * h_fft)
        cudafft.ifft(cuda_dict['x_fft'], cuda_dict['x'],
                     cuda_dict['ifft_plan'], False)
        x = np.array(cuda_dict['x'].get(), dtype=x.dtype, subok=True,
                     copy=False)
    return x


###############################################################################
# FFT Resampling

def setup_cuda_fft_resample(n_jobs, W, new_len):
    """Set up CUDA FFT resampling.

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
            x : instance of gpuarray
                Empty allocated GPU space for the data to resample.
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
    n_fft_x, n_fft_y = len(W), new_len
    cuda_fft_len_x = int((n_fft_x - (n_fft_x % 2)) // 2 + 1)
    cuda_fft_len_y = int((n_fft_y - (n_fft_y % 2)) // 2 + 1)
    if n_jobs == 'cuda':
        n_jobs = 1
        init_cuda()
        if _cuda_capable:
            # try setting up for float64
            from pycuda import gpuarray
            cudafft = _get_cudafft()
            try:
                # do the IFFT normalization now so we don't have to later
                W = gpuarray.to_gpu(W[:cuda_fft_len_x]
                                    .astype('complex_') / n_fft_y)
                cuda_dict.update(
                    use_cuda=True,
                    fft_plan=cudafft.Plan(n_fft_x, np.float64, np.complex128),
                    ifft_plan=cudafft.Plan(n_fft_y, np.complex128, np.float64),
                    x_fft=gpuarray.zeros(max(cuda_fft_len_x,
                                             cuda_fft_len_y), np.complex128),
                    x=gpuarray.empty(max(int(n_fft_x),
                                     int(n_fft_y)), np.float64))
                logger.info('Using CUDA for FFT resampling')
            except Exception:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
        else:
            logger.info('CUDA not used, CUDA could not be initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict, W


def fft_resample(x, W, new_len, npads, to_removes,
                 cuda_dict=dict(use_cuda=False)):
    """Do FFT resampling with a filter function (possibly using CUDA).

    Parameters
    ----------
    x : 1-d array
        The array to resample. Will be converted to float64 if necessary.
    W : 1-d array or gpuarray
        The filtering function to apply.
    new_len : int
        The size of the output array (before removing padding).
    npads : tuple of int
        Amount of padding to apply to the start and end of the
        signal before resampling.
    to_removes : tuple of int
        Number of samples to remove after resampling.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # add some padding at beginning and end to make this work a little cleaner
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    x = _smart_pad(x, npads)
    old_len = len(x)
    shorter = new_len < old_len
    if not cuda_dict['use_cuda']:
        N = int(min(new_len, old_len))
        # The below is equivalent to this, but faster
        # sl_1 = slice((N + 1) // 2)
        # y_fft = np.zeros(new_len, np.complex128)
        # x_fft = fft(x).ravel() * W
        # y_fft[sl_1] = x_fft[sl_1]
        # sl_2 = slice(-(N - 1) // 2, None)
        # y_fft[sl_2] = x_fft[sl_2]
        # y = np.real(ifft(y_fft, overwrite_x=True)).ravel()
        x_fft = rfft(x).ravel()
        x_fft *= W[np.arange(1, len(x) + 1) // 2].real
        y_fft = np.zeros(new_len, np.float64)
        sl_1 = slice(N)
        y_fft[sl_1] = x_fft[sl_1]
        if min(new_len, old_len) % 2 == 0:
            if new_len > old_len:
                y_fft[N - 1] /= 2.
        y = irfft(y_fft, overwrite_x=True).ravel()
    else:
        cudafft = _get_cudafft()
        cuda_dict['x'].set(np.concatenate((x, np.zeros(max(new_len - old_len,
                                                           0), x.dtype))))
        # do the fourier-domain operations, results put in second param
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        _multiply_inplace_c128(W, cuda_dict['x_fft'])
        # This is not straightforward, but because x_fft and y_fft share
        # the same data (and only one half of the full DFT is stored), we
        # don't have to transfer the slice like we do in scipy. All we
        # need to worry about is the Nyquist component, either halving it
        # or taking just the real component...
        use_len = new_len if shorter else old_len
        func = _real_c128 if shorter else _halve_c128
        if use_len % 2 == 0:
            nyq = int((use_len - (use_len % 2)) // 2)
            func(cuda_dict['x_fft'], slice=slice(nyq, nyq + 1))
        cudafft.ifft(cuda_dict['x_fft'], cuda_dict['x'],
                     cuda_dict['ifft_plan'], scale=False)
        y = cuda_dict['x'].get()[:new_len if shorter else None]

    # now let's trim it back to the correct size (if there was padding)
    if (to_removes > 0).any():
        keep = np.ones((new_len), dtype='bool')
        keep[:to_removes[0]] = False
        keep[-to_removes[1]:] = False
        y = np.compress(keep, y)

    return y


###############################################################################
# Misc

# this has to go in mne.cuda instead of mne.filter to avoid import errors
def _smart_pad(x, n_pad):
    """Pad vector x."""
    if (n_pad == 0).all():
        return x
    elif (n_pad < 0).any():
        raise RuntimeError('n_pad must be non-negative')
    # need to pad with zeros if len(x) <= npad
    l_z_pad = np.zeros(max(n_pad[0] - len(x) + 1, 0), dtype=x.dtype)
    r_z_pad = np.zeros(max(n_pad[0] - len(x) + 1, 0), dtype=x.dtype)
    return np.concatenate([l_z_pad, 2 * x[0] - x[n_pad[0]:0:-1], x,
                           2 * x[-1] - x[-2:-n_pad[1] - 2:-1], r_z_pad])
