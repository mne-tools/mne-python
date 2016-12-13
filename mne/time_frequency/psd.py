# Authors : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
#           Denis A. Engemann <denis.engemann@gmail.com>
# License : BSD 3-clause

import numpy as np
from ..externals.six import string_types

from scipy import fftpack
from scipy.signal import signaltools
from scipy.signal.windows import get_window

from ..parallel import parallel_func
from ..io.pick import _pick_data_channels
from ..utils import logger, verbose, _time_mask, get_reduction
from .multitaper import _psd_multitaper


def _pwelch(epoch, noverlap, nfft, fs, freq_mask, padding):
    """Aux function."""
    return _spectral_helper(epoch, fs, nperseg=nfft, noverlap=noverlap,
                    nfft=padding)[2][..., freq_mask, :]


def _check_nfft(n, n_fft, n_overlap):
    """Helper to make sure n_fft and n_overlap make sense."""
    n_fft = n if n_fft > n else n_fft
    n_overlap = n_fft - 1 if n_overlap >= n_fft else n_overlap
    return n_fft, n_overlap


def _check_psd_data(inst, tmin, tmax, picks, proj):
    """Helper to do checks on PSD data / pull arrays from inst."""
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    if not isinstance(inst, (BaseEpochs, BaseRaw, Evoked)):
        raise ValueError('epochs must be an instance of Epochs, Raw, or'
                         'Evoked. Got type {0}'.format(type(inst)))

    time_mask = _time_mask(inst.times, tmin, tmax, sfreq=inst.info['sfreq'])
    if picks is None:
        picks = _pick_data_channels(inst.info, with_ref_meg=False)
    if proj:
        # Copy first so it's not modified
        inst = inst.copy().apply_proj()

    sfreq = inst.info['sfreq']
    if isinstance(inst, BaseRaw):
        start, stop = np.where(time_mask)[0][[0, -1]]
        data, times = inst[picks, start:(stop + 1)]
    elif isinstance(inst, BaseEpochs):
        data = inst.get_data()[:, picks][:, :, time_mask]
    elif isinstance(inst, Evoked):
        data = inst.data[picks][:, time_mask]

    return data, sfreq


def _psd_welch(x, sfreq, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
               padding=None, reduction='mean', n_jobs=1):
    """Compute power spectral density (PSD) using Welch's method.

    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
    padding : int | None
        Number of samples to zero-pad the signal to. The default is None, which
        does not perform padding. padding has to be > n_fft
    reduction : str | float | callable | None
        The type of reduction to perform on windows. If string it can be 'mean'
        or 'median'. If float it is understood as % of values to trim before
        performing mean (has to be > 0 and < 0.5) ie. trimmed-mean. If function
        it has to perform the reduction along last dimension. If None - no
        reduction is performed and psd's for individual windows are returned
        (so that the output psds are of shape (windows, channels, frequencies)
        for 2d input data and (windows, epochs, channels, frequencies) for 3d
        input data)
    n_jobs : int
        Number of CPUs to use in the computation.

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or
        The power spectral densities. All dimensions up to the last will
        be the same as input.
    freqs : ndarray, shape (n_freqs,)
        The frequencies.
    """

    dshape = x.shape[:-1]
    n_times = x.shape[-1]
    x = x.reshape(-1, n_times)

    # Prep the PSD
    n_fft, n_overlap = _check_nfft(n_times, n_fft, n_overlap)
    if padding is not None and padding < n_fft:
        raise ValueError('padding must be more that n_fft.')
    win_size = n_fft / float(sfreq)
    logger.info("Effective window size : %0.3f (s)" % win_size)
    pnts = padding if padding is not None else n_fft
    freqs = np.arange(pnts // 2 + 1, dtype=float) * (sfreq / pnts)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Parallelize across first N-1 dimensions
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs)
    x_splits = np.array_split(x, n_jobs)
    f_psd = parallel(my_pwelch(d, noverlap=n_overlap, nfft=n_fft,
                     fs=sfreq, freq_mask=freq_mask,
                     padding=padding)
                     for d in x_splits)

    # Combining/reshaping to original data shape
    psds = np.concatenate(f_psd, axis=0)
    n_windows = psds.shape[-1]
    if reduction is not None:
        red_fun = get_reduction(reduction)
        psds = red_fun(psds).reshape(np.hstack([dshape, -1]))
    else:
        psds = psds.reshape(np.hstack([dshape, -1, n_windows]))
        psds = np.moveaxis(psds, -1, 0)

    return psds, freqs


@verbose
def psd_welch(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None, n_fft=256,
              n_overlap=0, padding=None, reduction='mean', picks=None,
              proj=False, n_jobs=1, verbose=None):
    """Compute the power spectral density (PSD) using Welch's method.

    Calculates periodigrams for a sliding window over the
    time dimension, then averages them together for each channel/epoch.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. If input is of type Raw,
        then psds will be shape (n_channels, n_freqs), if input is type Epochs
        then psds will be shape (n_epochs, n_channels, n_freqs).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_multitaper,
    csd_epochs

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return _psd_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,
                      n_overlap=n_overlap, padding=padding,
                      reduction=reduction, n_jobs=n_jobs)


@verbose
def psd_multitaper(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                   bandwidth=None, adaptive=False, low_bias=True,
                   normalization='length', picks=None, proj=False,
                   n_jobs=1, verbose=None):
    """Compute the power spectral density (PSD) using multitapers.

    Calculates spectral density for orthogonal tapers, then averages them
    together for each channel/epoch. See [1] for a description of the tapers
    and [2] for the general method.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation.
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. If input is of type Raw,
        then psds will be shape (n_channels, n_freqs), if input is type Epochs
        then psds will be shape (n_epochs, n_channels, n_freqs).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    References
    ----------
    .. [1] Slepian, D. "Prolate spheroidal wave functions, Fourier analysis,
           and uncertainty V: The discrete case." Bell System Technical
           Journal, vol. 57, 1978.

    .. [2] Percival D.B. and Walden A.T. "Spectral Analysis for Physical
           Applications: Multitaper and Conventional Univariate Techniques."
           Cambridge University Press, 1993.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_welch, csd_epochs

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return _psd_multitaper(data, sfreq, fmin=fmin, fmax=fmax,
                           bandwidth=bandwidth, adaptive=adaptive,
                           low_bias=low_bias,
                           normalization=normalization,  n_jobs=n_jobs)


def _spectral_helper(x, fs, window='hann', nperseg=256,
                    noverlap=None, nfft=None, detrend='constant',
                    scaling='density', axis=-1):
    '''will need to copy the docstring here'''
    axis = int(axis)

    # Ensure we have np.arrays
    x = np.asarray(x)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    if x.ndim > 1:
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))

    # test nperseg
    if x.shape[-1] < nperseg:
        warnings.warn('nperseg = {0:d}, is greater than input length = {1:d}, '
                      'using nperseg = {1:d}'.format(nperseg, x.shape[-1]))
        nperseg = x.shape[-1]

    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    elif noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    else:
        noverlap = int(noverlap)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend

    if isinstance(window, string_types) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if nfft % 2:
        num_freqs = (nfft + 1)//2
    else:
        num_freqs = nfft//2 + 1


    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft)
    result = result[..., :num_freqs]
    freqs = fftpack.fftfreq(nfft, 1/fs)[:num_freqs]

    result = (np.conjugate(result) * result).real
    result *= scale

    if nfft % 2:
        result[...,1:] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        result[...,1:-1] *= 2

    t = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1,
                  nperseg - noverlap) / float(fs)

    if not nfft % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # Output is going to have new last axis for window index
    if axis != -1:
        # Specify as positive axis index
        if axis < 0:
            axis = len(result.shape)-1-axis

        # Roll frequency axis back to axis where the data came from
        result = np.rollaxis(result, -1, axis)
    else:
        # Make sure window/time index is last axis
        result = np.rollaxis(result, -1, -2)

    return freqs, t, result


# fft helper
def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft):
    """
    Calculate windowed FFT, for internal use by scipy.signal._spectral_helper
    This is a helper function that does the main FFT calculation for
    _spectral helper. All input valdiation is performed there, and the data
    axis is assumed to be the last axis of x. It is not designed to be called
    externally. The windows are not averaged over; the result from each window
    is returned.
    Returns
    -------
    result : ndarray
        Array of FFT data
    References
    ----------
    .. [1] Stack Overflow, "Repeat NumPy array without replicating data?",
        http://stackoverflow.com/a/5568169
    Notes
    -----
    Copied from scipy.signal.spectral which was adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    result = fftpack.fft(result, n=nfft)

    return result
