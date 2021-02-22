from math import ceil
from copy import deepcopy
import numpy as np

from ..io.pick import _pick_data_channels, pick_info
from ..fixes import _import_fft
from ..utils import logger, verbose, deprecated, fill_doc, _validate_type
from .tfr import AverageTFR, _get_data


@fill_doc
def tfr_stft(inst, window_size, step_size=None, average=True, verbose=None):
    """Compute STFT Short-Term Fourier Transform using a sine window.

    Same computation as `~mne.time_frequency.tfr_array_stft`, but operates
    on `~mne.Epochs`, `~mne.Evoked` or `~mne.io.Raw` objects instead of
    :class:`NumPy arrays <numpy.ndarray>`.

    Parameters
    ----------
    inst : Epochs | Evoked | Raw
        The epochs or evoked object.
    window_size : int
        Length of the STFT window in samples (must be a multiple of 4).
    step_size : int
        Step between successive windows in samples (must be a multiple of 2,
        a divider of ``window_size`` and smaller than ``window_size``/2)
        (default: window_size/2).
    %(tfr_average)s
    %(verbose)s

    Returns
    -------
    power : AverageTFR
        The averaged power.
    """
    from ..epochs import BaseEpochs
    from .tfr import EpochsTFR

    # verbose dec is used b/c subfunctions are verbose
    data = _get_data(inst, False)
    picks = _pick_data_channels(inst.info)
    info = pick_info(inst.info, picks)
    data = data[:, picks, :]

    # compute STFT on the numpy array
    power, freqs, times = tfr_array_stft(
        data, sfreq=info['sfreq'], window_size=window_size,
        step_size=step_size, return_times=True,
        verbose=verbose)

    if average:
        # how many epochs are we averaging
        nave = len(data)
        out = AverageTFR(info, power, times, freqs, nave, method='stft-power')
    else:
        if isinstance(inst, BaseEpochs):
            meta = deepcopy(inst._metadata)
            evs = deepcopy(inst.events)
            ev_id = deepcopy(inst.event_id)
        else:
            # if the input is of class Evoked
            meta = evs = ev_id = None
        out = EpochsTFR(info, power, times, freqs, method='stft-power',
                        events=evs, event_id=ev_id, metadata=meta)

    return out


def tfr_array_stft(data, sfreq, window_size, step_size=None,
                   return_times=False, verbose=True):
    """STFT Short-Term Fourier Transform using a sine window.

    The transformation is designed to be a tight frame that can be
    perfectly inverted. It only returns the positive frequencies.

    Parameters
    ----------
    x : array, shape (n_epochs, n_channels, n_times)
        Containing multi-channels signal.
    window_size : int
        Length of the STFT window in samples (must be a multiple of 4).
    step_size : int
        Step between successive windows in samples (must be a multiple of 2,
        a divider of ``window_size`` and smaller than ``window_size``/2)
        (default: window_size/2).
    return_times : bool
        Whether to return the time points of the STFT time-frequency
        representation. Default is False.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    X : array, shape (n_channels, wsize // 2 + 1, n_step)
        STFT coefficients for positive frequencies with
        ``n_step = ceil(T / tstep)``.
    freqs : array, shape (wsize // 2 + 1,)
        STFT frequencies corresponding.

    See Also
    --------
    istft
    stftfreq
    """
    n_epochs, n_channels, n_signals = data.shape

    power_list = []

    for epochidx in range(n_epochs):
        # compute STFT
        power = _stft(data, window_size=window_size, step_size=step_size,
                      verbose=verbose)
        power_list.append(power)
    power_list = np.array(power_list)

    # compute frequencies associated with the STFT
    freqs = stftfreq(wsize=window_size, sfreq=sfreq)

    # compute time points in seconds
    times = _stft_times(n_signals=n_signals, window_size=window_size,
                        step_size=step_size, sfreq=sfreq)
    if return_times:
        return power_list, freqs, times
    return power_list, freqs


@deprecated('This function call for STFT is deprecated. Use instead '
            '"tfr_array_stft", or "tfr_stft".')
@verbose
def stft(x, wsize, tstep=None, verbose=None):
    """STFT Short-Term Fourier Transform using a sine window.

    The transformation is designed to be a tight frame that can be
    perfectly inverted. It only returns the positive frequencies.

    Parameters
    ----------
    x : array, shape (n_signals, n_times)
        Containing multi-channels signal.
    wsize : int
        Length of the STFT window in samples (must be a multiple of 4).
    tstep : int
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    %(verbose)s

    Returns
    -------
    X : array, shape (n_signals, wsize // 2 + 1, n_step)
        STFT coefficients for positive frequencies with
        ``n_step = ceil(T / tstep)``.

    See Also
    --------
    istft
    stftfreq
    """
    return _stft(x, window_size=wsize, step_size=tstep, verbose=verbose)


def _stft_times(n_signals, window_size, step_size, sfreq):
    """Compute time points in seconds of STFT."""
    # Creates a [n,2] array that holds the sample range of each window that
    # is used to index the raw data for a sliding window analysis
    samples_start = np.arange(0, n_signals - window_size + 1.0, step_size).astype(int)
    samples_end = np.arange(window_size, n_signals + 1, step_size).astype(int)

    # compute window endpoints in samples and frequencies
    samples_wins = np.append(
        samples_start[:, np.newaxis], samples_end[:, np.newaxis], axis=1
    )
    second_wins = np.divide(samples_wins, sfreq)
    second_points = np.mean(second_wins, axis=1)
    return second_points


def _stft(data, window_size, step_size, verbose):
    rfft = _import_fft('rfft')
    _validate_type(data, np.ndarray, 'data')
    if not np.isrealobj(data):
        raise ValueError("x is not a real valued array")

    if data.ndim == 1:
        data = data[None, :]

    # XXX: do we want this?
    # if data.ndim != 3:
    #     raise ValueError(
    #         'data must be 3D with shape (n_epochs, n_channels, n_times), '
    #         f'got {data.shape}')

    # get the data shape
    n_signals, T = data.shape
    window_size = int(window_size)

    # Errors and warnings
    if window_size % 4:
        raise ValueError('The window length must be a multiple of 4.')

    if step_size is None:
        step_size = window_size / 2

    step_size = int(step_size)

    if (window_size % step_size) or (step_size % 2):
        raise ValueError('The step size must be a multiple of 2 and a '
                         'divider of the window length.')

    if step_size > window_size / 2:
        raise ValueError('The step size must be smaller than half the '
                         'window length.')

    n_step = int(ceil(T / float(step_size)))
    n_freq = window_size // 2 + 1
    logger.info("Number of frequencies: %d" % n_freq)
    logger.info("Number of time steps: %d" % n_step)

    X = np.zeros((n_signals, n_freq, n_step), dtype=np.complex128)

    if n_signals == 0:
        return X

    # Defining sine window
    win = np.sin(np.arange(.5, window_size + .5) / window_size * np.pi)
    win2 = win ** 2

    swin = np.zeros((n_step - 1) * step_size + window_size)
    for t in range(n_step):
        swin[t * step_size:t * step_size + window_size] += win2
    swin = np.sqrt(window_size * swin)

    # Zero-padding and Pre-processing for edges
    xp = np.zeros((n_signals, window_size + (n_step - 1) * step_size),
                  dtype=data.dtype)
    xp[:, (window_size - step_size) // 2: (window_size - step_size) // 2 + T] = data
    data = xp

    for t in range(n_step):
        # Framing
        wwin = win / swin[t * step_size: t * step_size + window_size]
        frame = data[:, t * step_size: t * step_size + window_size] * wwin[None, :]
        # FFT
        X[:, :, t] = rfft(frame)

    return X


def istft(X, tstep=None, Tx=None):
    """ISTFT Inverse Short-Term Fourier Transform using a sine window.

    Parameters
    ----------
    X : array, shape (..., wsize / 2 + 1, n_step)
        The STFT coefficients for positive frequencies.
    tstep : int
        Step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) (default: wsize/2).
    Tx : int
        Length of returned signal. If None Tx = n_step * tstep.

    Returns
    -------
    x : array, shape (Tx,)
        Array containing the inverse STFT signal.

    See Also
    --------
    stft
    """
    # Errors and warnings
    irfft = _import_fft('irfft')
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError(f'X must have ndim >= 2, got {X.ndim}')
    n_win, n_step = X.shape[-2:]
    signal_shape = X.shape[:-2]
    if n_win % 2 == 0:
        raise ValueError('The number of rows of the STFT matrix must be odd.')

    wsize = 2 * (n_win - 1)
    if tstep is None:
        tstep = wsize / 2

    if wsize % tstep:
        raise ValueError('The step size must be a divider of two times the '
                         'number of rows of the STFT matrix minus two.')

    if wsize % 2:
        raise ValueError('The step size must be a multiple of 2.')

    if tstep > wsize / 2:
        raise ValueError('The step size must be smaller than the number of '
                         'rows of the STFT matrix minus one.')

    if Tx is None:
        Tx = n_step * tstep

    T = n_step * tstep

    x = np.zeros(signal_shape + (T + wsize - tstep,), dtype=np.float64)

    if np.prod(signal_shape) == 0:
        return x[..., :Tx]

    # Defining sine window
    win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
    # win = win / norm(win);

    # Pre-processing for edges
    swin = np.zeros(T + wsize - tstep, dtype=np.float64)
    for t in range(n_step):
        swin[t * tstep:t * tstep + wsize] += win ** 2
    swin = np.sqrt(swin / wsize)

    for t in range(n_step):
        # IFFT
        frame = irfft(X[..., t], wsize)
        # Overlap-add
        frame *= win / swin[t * tstep:t * tstep + wsize]
        x[..., t * tstep: t * tstep + wsize] += frame

    # Truncation
    x = x[..., (wsize - tstep) // 2: (wsize - tstep) // 2 + T + 1]
    x = x[..., :Tx].copy()
    return x


def stftfreq(wsize, sfreq=None):  # noqa: D401
    """Compute frequencies of stft transformation.

    Parameters
    ----------
    wsize : int
        Size of stft window.
    sfreq : float
        Sampling frequency. If None the frequencies are given between 0 and pi
        otherwise it's given in Hz.

    Returns
    -------
    freqs : array
        The positive frequencies returned by stft.

    See Also
    --------
    stft
    istft
    """
    rfftfreq = _import_fft('rfftfreq')
    freqs = rfftfreq(wsize)
    if sfreq is not None:
        freqs *= float(sfreq)
    return freqs


def stft_norm2(X):
    """Compute L2 norm of STFT transform.

    It takes into account that stft only return positive frequencies.
    As we use tight frame this quantity is conserved by the stft.

    Parameters
    ----------
    X : 3D complex array
        The STFT transforms

    Returns
    -------
    norms2 : array
        The squared L2 norm of every row of X.
    """
    X2 = (X * X.conj()).real
    # compute all L2 coefs and remove first and last frequency once.
    norms2 = (2. * X2.sum(axis=2).sum(axis=1) - np.sum(X2[:, 0, :], axis=1) -
              np.sum(X2[:, -1, :], axis=1))
    return norms2


def stft_norm1(X):
    """Compute L1 norm of STFT transform.

    It takes into account that stft only return positive frequencies.

    Parameters
    ----------
    X : 3D complex array
        The STFT transforms

    Returns
    -------
    norms : array
        The L1 norm of every row of X.
    """
    X_abs = np.abs(X)
    # compute all L1 coefs and remove first and last frequency once.
    norms = (2. * X_abs.sum(axis=(1, 2)) -
             np.sum(X_abs[:, 0, :], axis=1) - np.sum(X_abs[:, -1, :], axis=1))
    return norms
