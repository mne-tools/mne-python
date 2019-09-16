# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .fixes import rfft, irfft
from .utils import (sizeof_fmt, logger, get_config, warn, _explain_exception,
                    verbose)


_cuda_capable = False


def get_cuda_memory(kind='available'):
    """Get the amount of free memory for CUDA operations.

    Parameters
    ----------
    kind : str
        Can be "available" or "total".

    Returns
    -------
    memory : str
        The amount of available or total memory as a human-readable string.
    """
    if not _cuda_capable:
        warn('CUDA not enabled, returning zero for memory')
        mem = 0
    else:
        import cupy
        mem = cupy.cuda.runtime.memGetInfo()[dict(available=0, total=1)[kind]]
    return sizeof_fmt(mem)


@verbose
def init_cuda(ignore_config=False, verbose=None):
    """Initialize CUDA functionality.

    This function attempts to load the necessary interfaces
    (hardware connectivity) to run CUDA-based filtering. This
    function should only need to be run once per session.

    If the config var (set via mne.set_config or in ENV)
    MNE_USE_CUDA == 'true', this function will be executed when
    the first CUDA setup is performed. If this variable is not
    set, this function can be manually executed.

    Parameters
    ----------
    ignore_config : bool
        If True, ignore the config value MNE_USE_CUDA and force init.
    %(verbose)s
    """
    global _cuda_capable
    if _cuda_capable:
        return
    if not ignore_config and (get_config('MNE_USE_CUDA', 'false').lower() !=
                              'true'):
        logger.info('CUDA not enabled in config, skipping initialization')
        return
    # Triage possible errors for informative messaging
    _cuda_capable = False
    try:
        import cupy  # noqa
    except ImportError:
        warn('module cupy not found, CUDA not enabled')
        return
    device_id = int(get_config('MNE_CUDA_DEVICE', '0'))
    try:
        # Initialize CUDA
        _set_cuda_device(device_id, verbose)
    except Exception:
        warn('so CUDA device could be initialized, likely a hardware error, '
             'CUDA not enabled%s' % _explain_exception())
        return

    _cuda_capable = True
    # Figure out limit for CUDA FFT calculations
    logger.info('Enabling CUDA with %s available memory' % get_cuda_memory())


@verbose
def set_cuda_device(device_id, verbose=None):
    """Set the CUDA device temporarily for the current session.

    Parameters
    ----------
    device_id : int
        Numeric ID of the CUDA-capable device you want MNE-Python to use.
    %(verbose)s
    """
    if _cuda_capable:
        _set_cuda_device(device_id, verbose)
    elif get_config('MNE_USE_CUDA', 'false').lower() == 'true':
        init_cuda()
        _set_cuda_device(device_id, verbose)
    else:
        warn('Could not set CUDA device because CUDA is not enabled; either '
             'run mne.cuda.init_cuda() first, or set the MNE_USE_CUDA config '
             'variable to "true".')


@verbose
def _set_cuda_device(device_id, verbose=None):
    """Set the CUDA device."""
    import cupy
    cupy.cuda.Device(device_id).use()
    logger.info('Now using CUDA device {}'.format(device_id))


###############################################################################
# Repeated FFT multiplication

def _setup_cuda_fft_multiply_repeated(n_jobs, h, n_fft,
                                      kind='FFT FIR filtering'):
    """Set up repeated CUDA FFT multiplication with a given filter.

    Parameters
    ----------
    n_jobs : int | str
        If n_jobs == 'cuda', the function will attempt to set up for CUDA
        FFT multiplication.
    h : array
        The filtering function that will be used repeatedly.
    n_fft : int
        The number of points in the FFT.
    kind : str
        The kind to report to the user.

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
        This will either be a gpuarray (if CUDA enabled) or ndarray.

    Notes
    -----
    This function is designed to be used with fft_multiply_repeated().
    """
    cuda_dict = dict(n_fft=n_fft, rfft=rfft, irfft=irfft,
                     h_fft=rfft(h, n=n_fft))
    if n_jobs == 'cuda':
        n_jobs = 1
        init_cuda()
        if _cuda_capable:
            import cupy
            try:
                # do the IFFT normalization now so we don't have to later
                h_fft = cupy.array(cuda_dict['h_fft'])
                logger.info('Using CUDA for %s' % kind)
            except Exception as exp:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large: "%s"), falling back to '
                            'n_jobs=1' % str(exp))
            cuda_dict.update(h_fft=h_fft,
                             rfft=_cuda_upload_rfft,
                             irfft=_cuda_irfft_get)
        else:
            logger.info('CUDA not used, CUDA could not be initialized, '
                        'falling back to n_jobs=1')
    return n_jobs, cuda_dict


def _fft_multiply_repeated(x, cuda_dict):
    """Do FFT multiplication by a filter function (possibly using CUDA).

    Parameters
    ----------
    h_fft : 1-d array or gpuarray
        The filtering array to apply.
    x : 1-d array
        The array to filter.
    n_fft : int
        The number of points in the FFT.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    # do the fourier-domain operations
    x_fft = cuda_dict['rfft'](x, cuda_dict['n_fft'])
    x_fft *= cuda_dict['h_fft']
    x = cuda_dict['irfft'](x_fft, cuda_dict['n_fft'])
    return x


###############################################################################
# FFT Resampling

def _setup_cuda_fft_resample(n_jobs, W, new_len):
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

    Notes
    -----
    This function is designed to be used with fft_resample().
    """
    cuda_dict = dict(use_cuda=False, rfft=rfft, irfft=irfft)
    rfft_len_x = len(W) // 2 + 1
    # fold the window onto inself (should be symmetric) and truncate
    W = W.copy()
    W[1:rfft_len_x] = (W[1:rfft_len_x] + W[::-1][:rfft_len_x - 1]) / 2.
    W = W[:rfft_len_x]
    if n_jobs == 'cuda':
        n_jobs = 1
        init_cuda()
        if _cuda_capable:
            try:
                import cupy
                # do the IFFT normalization now so we don't have to later
                W = cupy.array(W)
                logger.info('Using CUDA for FFT resampling')
            except Exception:
                logger.info('CUDA not used, could not instantiate memory '
                            '(arrays may be too large), falling back to '
                            'n_jobs=1')
            else:
                cuda_dict.update(use_cuda=True,
                                 rfft=_cuda_upload_rfft,
                                 irfft=_cuda_irfft_get)
        else:
            logger.info('CUDA not used, CUDA could not be initialized, '
                        'falling back to n_jobs=1')
    cuda_dict['W'] = W
    return n_jobs, cuda_dict


def _cuda_upload_rfft(x, n, axis=-1):
    """Upload and compute rfft."""
    import cupy
    return cupy.fft.rfft(cupy.array(x), n=n, axis=axis)


def _cuda_irfft_get(x, n, axis=-1):
    """Compute irfft and get."""
    import cupy
    return cupy.fft.irfft(x, n=n, axis=axis).get()


def _fft_resample(x, new_len, npads, to_removes, cuda_dict=None,
                  pad='reflect_limited'):
    """Do FFT resampling with a filter function (possibly using CUDA).

    Parameters
    ----------
    x : 1-d array
        The array to resample. Will be converted to float64 if necessary.
    new_len : int
        The size of the output array (before removing padding).
    npads : tuple of int
        Amount of padding to apply to the start and end of the
        signal before resampling.
    to_removes : tuple of int
        Number of samples to remove after resampling.
    cuda_dict : dict
        Dictionary constructed using setup_cuda_multiply_repeated().
    pad : str
        The type of padding to use. Supports all :func:`np.pad` ``mode``
        options. Can also be "reflect_limited" (default), which pads with a
        reflected version of each vector mirrored on the first and last values
        of the vector, followed by zeros.

        .. versionadded:: 0.15

    Returns
    -------
    x : 1-d array
        Filtered version of x.
    """
    cuda_dict = dict(use_cuda=False) if cuda_dict is None else cuda_dict
    # add some padding at beginning and end to make this work a little cleaner
    if x.dtype != np.float64:
        x = x.astype(np.float64)
    x = _smart_pad(x, npads, pad)
    old_len = len(x)
    shorter = new_len < old_len
    use_len = new_len if shorter else old_len
    x_fft = cuda_dict['rfft'](x, None)
    if use_len % 2 == 0:
        nyq = use_len // 2
        x_fft[nyq:nyq + 1] *= 2 if shorter else 0.5
    x_fft *= cuda_dict['W']
    y = cuda_dict['irfft'](x_fft, new_len)

    # now let's trim it back to the correct size (if there was padding)
    if (to_removes > 0).any():
        y = y[to_removes[0]:y.shape[0] - to_removes[1]]

    return y


###############################################################################
# Misc

# this has to go in mne.cuda instead of mne.filter to avoid import errors
def _smart_pad(x, n_pad, pad='reflect_limited'):
    """Pad vector x."""
    n_pad = np.asarray(n_pad)
    assert n_pad.shape == (2,)
    if (n_pad == 0).all():
        return x
    elif (n_pad < 0).any():
        raise RuntimeError('n_pad must be non-negative')
    if pad == 'reflect_limited':
        # need to pad with zeros if len(x) <= npad
        l_z_pad = np.zeros(max(n_pad[0] - len(x) + 1, 0), dtype=x.dtype)
        r_z_pad = np.zeros(max(n_pad[1] - len(x) + 1, 0), dtype=x.dtype)
        return np.concatenate([l_z_pad, 2 * x[0] - x[n_pad[0]:0:-1], x,
                               2 * x[-1] - x[-2:-n_pad[1] - 2:-1], r_z_pad])
    else:
        return np.pad(x, (tuple(n_pad),), pad)
