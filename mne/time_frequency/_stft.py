from math import ceil
import numpy as np

from ..fixes import fft, ifft, fftfreq
from ..utils import logger, verbose


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
    if not np.isrealobj(x):
        raise ValueError("x is not a real valued array")

    if x.ndim == 1:
        x = x[None, :]

    n_signals, T = x.shape
    wsize = int(wsize)

    # Errors and warnings
    if wsize % 4:
        raise ValueError('The window length must be a multiple of 4.')

    if tstep is None:
        tstep = wsize / 2

    tstep = int(tstep)

    if (wsize % tstep) or (tstep % 2):
        raise ValueError('The step size must be a multiple of 2 and a '
                         'divider of the window length.')

    if tstep > wsize / 2:
        raise ValueError('The step size must be smaller than half the '
                         'window length.')

    n_step = int(ceil(T / float(tstep)))
    n_freq = wsize // 2 + 1
    logger.info("Number of frequencies: %d" % n_freq)
    logger.info("Number of time steps: %d" % n_step)

    X = np.zeros((n_signals, n_freq, n_step), dtype=np.complex128)

    if n_signals == 0:
        return X

    # Defining sine window
    win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
    win2 = win ** 2

    swin = np.zeros((n_step - 1) * tstep + wsize)
    for t in range(n_step):
        swin[t * tstep:t * tstep + wsize] += win2
    swin = np.sqrt(wsize * swin)

    # Zero-padding and Pre-processing for edges
    xp = np.zeros((n_signals, wsize + (n_step - 1) * tstep),
                  dtype=x.dtype)
    xp[:, (wsize - tstep) // 2: (wsize - tstep) // 2 + T] = x
    x = xp

    for t in range(n_step):
        # Framing
        wwin = win / swin[t * tstep: t * tstep + wsize]
        frame = x[:, t * tstep: t * tstep + wsize] * wwin[None, :]
        # FFT
        fframe = fft(frame)
        X[:, :, t] = fframe[:, :n_freq]

    return X


def istft(X, tstep=None, Tx=None):
    """ISTFT Inverse Short-Term Fourier Transform using a sine window.

    Parameters
    ----------
    X : array, shape (n_signals, wsize / 2 + 1, n_step)
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
    n_signals, n_win, n_step = X.shape
    if (n_win % 2 == 0):
        ValueError('The number of rows of the STFT matrix must be odd.')

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

    x = np.zeros((n_signals, T + wsize - tstep), dtype=np.float64)

    if n_signals == 0:
        return x[:, :Tx]

    # Defining sine window
    win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
    # win = win / norm(win);

    # Pre-processing for edges
    swin = np.zeros(T + wsize - tstep, dtype=np.float64)
    for t in range(n_step):
        swin[t * tstep:t * tstep + wsize] += win ** 2
    swin = np.sqrt(swin / wsize)

    fframe = np.empty((n_signals, n_win + wsize // 2 - 1), dtype=X.dtype)
    for t in range(n_step):
        # IFFT
        fframe[:, :n_win] = X[:, :, t]
        fframe[:, n_win:] = np.conj(X[:, wsize // 2 - 1: 0: -1, t])
        frame = ifft(fframe)
        wwin = win / swin[t * tstep:t * tstep + wsize]
        # Overlap-add
        x[:, t * tstep: t * tstep + wsize] += np.real(np.conj(frame) * wwin)

    # Truncation
    x = x[:, (wsize - tstep) // 2: (wsize - tstep) // 2 + T + 1][:, :Tx].copy()
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
    n_freq = wsize // 2 + 1
    freqs = fftfreq(wsize)
    freqs = np.abs(freqs[:n_freq])
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
