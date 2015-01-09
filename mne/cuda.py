# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.fftpack import fft, ifft
try:
    import pycuda.gpuarray as gpuarray
    from pycuda.driver import mem_get_info
    from scikits.cuda import fft as cudafft
except (ImportError, OSError):
    # need OSError because scikits.cuda throws it if cufft not found
    pass

from .parallel import check_n_jobs
from .utils import sizeof_fmt, logger


# Support CUDA for FFTs; requires scikits.cuda and pycuda
cuda_capable = False
cuda_multiply_inplace_c128 = cuda_halve_c128 = cuda_real_c128 = None


def init_cuda():
    """Initialize CUDA functionality

    This function attempts to load the necessary interfaces
    (hardware connectivity) to run CUDA-based filtering. This
    function should only need to be run once per session.

    If the config var (set via mne.set_config or in ENV)
    MNE_USE_CUDA == 'true', this function will be executed when
    importing mne. If this variable is not set, this function can
    be manually executed.
    """
    global cuda_capable, cuda_multiply_inplace_c128, cuda_halve_c128
    global cuda_real_c128
    if cuda_capable is True:
        logger.info('CUDA previously enabled, currently %s available memory'
                    % sizeof_fmt(mem_get_info()[0]))
        return
    # Triage possible errors for informative messaging
    cuda_capable = False
    try:
        import pycuda.gpuarray
        import pycuda.driver
    except ImportError as exp:
        logger.warning('module pycuda not found, CUDA not enabled (%s)'
                       % exp)
        return
    try:
        # Initialize CUDA; happens with importing autoinit
        import pycuda.autoinit  # noqa, analysis:ignore
    except ImportError as exp:
        logger.warning('pycuda.autoinit could not be imported, likely '
                       'a hardware error, CUDA not enabled (%s)' % exp)
        return
    # Make sure scikits.cuda is installed
    try:
        from scikits.cuda import fft as cudafft
    except ImportError as exp:
        logger.warning('module scikits.cuda not found, CUDA not '
                       'enabled (%s)' % exp)
        return

    # Make our multiply inplace kernel
    from pycuda.elementwise import ElementwiseKernel

    # let's construct our own CUDA multiply in-place function
    cuda_multiply_inplace_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a, pycuda::complex<double> *b',
        'b[i] *= a[i]', 'multiply_inplace')
    cuda_halve_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a', 'a[i] /= 2.0', 'halve_value')
    cuda_real_c128 = ElementwiseKernel(
        'pycuda::complex<double> *a', 'a[i] = real(a[i])', 'real_value')

    # Make sure we can use 64-bit FFTs
    try:
        cudafft.Plan(16, np.float64, np.complex128)  # will get auto-GC'ed
    except:
        logger.warning('Device does not support 64-bit FFTs, '
                       'CUDA not enabled')
        return
    cuda_capable = True
    # Figure out limit for CUDA FFT calculations
    logger.info('Enabling CUDA with %s available memory'
                % sizeof_fmt(mem_get_info()[0]))


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
                     x_fft=None, x=None)
    n_fft = len(h_fft)
    cuda_fft_len = int((n_fft - (n_fft % 2)) / 2 + 1)
    n_jobs = check_n_jobs(n_jobs, allow_cuda=True)
    if n_jobs == 'cuda':
        n_jobs = 1
        if cuda_capable:
            # set up all arrays necessary for CUDA
            # try setting up for float64
            try:
                # do the IFFT normalization now so we don't have to later
                h_fft = gpuarray.to_gpu(h_fft[:cuda_fft_len].astype('complex_')
                                        / len(h_fft))
                cuda_dict.update(
                    use_cuda=True,
                    fft_plan=cudafft.Plan(n_fft, np.float64, np.complex128),
                    ifft_plan=cudafft.Plan(n_fft, np.complex128, np.float64),
                    x_fft=gpuarray.empty(cuda_fft_len, np.complex128),
                    x=gpuarray.empty(int(n_fft), np.float64))
                logger.info('Using CUDA for FFT FIR filtering')
            except Exception:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
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
        cuda_dict['x'].set(x.astype(np.float64))
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        cuda_multiply_inplace_c128(h_fft, cuda_dict['x_fft'])
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
    n_jobs = check_n_jobs(n_jobs, allow_cuda=True)
    if n_jobs == 'cuda':
        n_jobs = 1
        if cuda_capable:
            # try setting up for float64
            try:
                # do the IFFT normalization now so we don't have to later
                W = gpuarray.to_gpu(W[:cuda_fft_len_x].astype('complex_')
                                    / n_fft_y)
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
    old_len = len(x)
    shorter = new_len < old_len
    if not cuda_dict['use_cuda']:
        N = int(min(new_len, old_len))
        sl_1 = slice((N + 1) // 2)
        y_fft = np.zeros(new_len, np.complex128)
        x_fft = fft(x).ravel() * W
        y_fft[sl_1] = x_fft[sl_1]
        sl_2 = slice(-(N - 1) // 2, None)
        y_fft[sl_2] = x_fft[sl_2]
        y = np.real(ifft(y_fft, overwrite_x=True)).ravel()
    else:
        cuda_dict['x'].set(np.concatenate((x, np.zeros(max(new_len - old_len,
                                                           0), x.dtype))))
        # do the fourier-domain operations, results put in second param
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        cuda_multiply_inplace_c128(W, cuda_dict['x_fft'])
        # This is not straightforward, but because x_fft and y_fft share
        # the same data (and only one half of the full DFT is stored), we
        # don't have to transfer the slice like we do in scipy. All we
        # need to worry about is the Nyquist component, either halving it
        # or taking just the real component...
        use_len = new_len if shorter else old_len
        func = cuda_real_c128 if shorter else cuda_halve_c128
        if use_len % 2 == 0:
            nyq = int((use_len - (use_len % 2)) // 2)
            func(cuda_dict['x_fft'], slice=slice(nyq, nyq + 1))
        cudafft.ifft(cuda_dict['x_fft'], cuda_dict['x'],
                     cuda_dict['ifft_plan'], scale=False)
        y = cuda_dict['x'].get()[:new_len if shorter else None]

    # now let's trim it back to the correct size (if there was padding)
    if to_remove > 0:
        keep = np.ones((new_len), dtype='bool')
        keep[:to_remove] = False
        keep[-to_remove:] = False
        y = np.compress(keep, y)

    return y


###############################################################################
# Stockwell transform

def setup_cuda_st_power_itc(n_jobs, W, n_trials):
    """Set up repeated CUDA FFT multiplication with a given filter

    Parameters
    ----------
    n_jobs : int | str
        If n_jobs == 'cuda', the function will attempt to set up for CUDA
        FFT multiplication.
    W : array
        The frequency-domain representation of the set of windows that will
        be used repeatedly. If n_jobs='cuda', this will be turned into a
        gpuarray.
    n_trials : int
        Number of trials to process.

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

    Notes
    -----
    This function is designed to be used with st_power_itc().
    """
    cuda_dict = dict(use_cuda=False, W=W)
    n_fft = W.shape[1]
    n_jobs = check_n_jobs(n_jobs, allow_cuda=True)
    if n_jobs == 'cuda':
        n_jobs = 1
        if cuda_capable:
            try:
                W = gpuarray.to_gpu(W.astype(np.complex128) / n_fft)
                fft_plan = cudafft.Plan(n_fft, np.complex128, np.complex128)
                ifft_plan = cudafft.Plan(n_fft, np.complex128, np.complex128)
                x_fft = gpuarray.empty((n_trials, n_fft), np.complex128)
                x = gpuarray.empty((n_trials, n_fft), np.complex128)
                xx = gpuarray.empty((n_trials, 2 * n_fft), np.complex128)
                YY = gpuarray.empty((n_trials, n_fft), np.complex128)
                ST = gpuarray.empty((n_trials, n_fft), np.complex128)
                cuda_dict.update(use_cuda=True, fft_plan=fft_plan, W=W,
                                 ifft_plan=ifft_plan, x_fft=x_fft, x=x, xx=xx,
                                 YY=YY, ST=ST)
                logger.info('Using CUDA for Stockwell calculations')
            except ImportError:  # XXX set back to Exception:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
        else:
            logger.info('CUDA not used, CUDA has not been initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict


def st_power_itc(x, start_f, compute_itc, zero_pad, decim, cuda_dict):
    """Aux function"""
    n_samp = x.shape[-1]
    n_out = (n_samp - zero_pad) // decim
    W = cuda_dict['W']  # window functions on the GPU
    psd = np.empty((len(W), n_out))
    itc = np.empty_like(psd) if compute_itc else None
    if cuda_dict['use_cuda']:
        cuda_dict['x'].set(x.astype(np.complex128))
        cudafft.fft(cuda_dict['x'], cuda_dict['x_fft'], cuda_dict['fft_plan'])
        # XXX this doesn't work; we'll need to adapt this:
        # http://hannes-brt.github.io/blog/2013/08/07/column-slicing-in-pycuda
        x_fft = cuda_dict['x_fft'].get()
        cuda_dict['xx'].set(np.concatenate([x_fft, x_fft], axis=-1))
        for i_f, window in enumerate(W):
            f = start_f + i_f
            # XXX should col slice here, too :(
            YY = np.ascontiguousarray(cuda_dict['xx'].get()[:, f:f + n_samp])
            cuda_dict['YY'].set(YY)
            cuda_dict['YY'] *= window  # on GPU
            cudafft.ifft(cuda_dict['YY'], cuda_dict['ST'],
                         cuda_dict['ifft_plan'])
            # could adapt page above for real col slicing, might be slower?
            TFR = cuda_dict['ST']
            TFR_abs = abs(TFR)
            # XXX should be able to use ExclusiveScanKernel, but
            # it won't compile (argh)
            # Also this could probably be combined nicely in a single-pass
            # function
            if compute_itc:
                TFR /= TFR_abs
                itc[i_f] = abs(TFR).get()[:, :-zero_pad:decim].mean(axis=0)
            TFR_abs *= TFR_abs
            psd[i_f] = TFR_abs.get()[:, :-zero_pad:decim].mean(axis=0)
    else:
        X = fft(x)
        XX = np.concatenate([X, X], axis=-1)
        for i_f, window in enumerate(W):
            f = start_f + i_f
            ST = ifft(XX[:, f:f + n_samp] * window)
            TFR = ST[:, :-zero_pad:decim]
            TFR_abs = np.abs(TFR)
            if compute_itc:
                TFR /= TFR_abs
                itc[i_f] = np.abs(np.mean(TFR, axis=0))
            TFR_abs *= TFR_abs
            psd[i_f] = np.mean(TFR_abs, axis=0)
    return psd, itc


###############################################################################
# Misc

# this has to go in mne.cuda instead of mne.filter to avoid import errors
def _smart_pad(x, n_pad):
    """Pad vector x
    """
    # need to pad with zeros if len(x) <= npad
    z_pad = np.zeros(max(n_pad - len(x) + 1, 0), dtype=x.dtype)
    return np.r_[z_pad, 2 * x[0] - x[n_pad:0:-1], x,
                 2 * x[-1] - x[-2:-n_pad - 2:-1], z_pad]
