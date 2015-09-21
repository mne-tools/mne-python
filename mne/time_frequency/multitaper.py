# Author : Martin Luessi mluessi@nmr.mgh.harvard.edu (2012)
# License : BSD 3-clause

# Parts of this code were copied from NiTime http://nipy.sourceforge.net/nitime
from warnings import warn

import numpy as np
from scipy import fftpack, linalg
import warnings

from ..parallel import parallel_func
from ..utils import verbose, sum_squared


def tridisolve(d, e, b, overwrite_b=True):
    """
    Symmetric tridiagonal system solver, from Golub and Van Loan pg 157

    Note: Copied from NiTime

    Parameters
    ----------

    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector

    Returns
    -------

    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b

    """
    N = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in range(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in range(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in range(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """Perform an inverse iteration to find the eigenvector corresponding
    to the given eigenvalue in a symmetric tridiagonal system.

    Note: Copied from NiTime

    Parameters
    ----------

    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates

    Returns
    -------

    e: ndarray
      The converged eigenvector

    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def dpss_windows(N, half_nbw, Kmax, low_bias=True, interp_from=None,
                 interp_kind='linear'):
    """
    Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N.

    Note: Copied from NiTime

    Parameters
    ----------
    N : int
        Sequence length
    half_nbw : float, unitless
        Standardized half bandwidth corresponding to 2 * half_bw = BW*f0
        = BW*N/dt but with dt taken as 1
    Kmax : int
        Number of DPSS windows to return is Kmax (orders 0 through Kmax-1)
    low_bias : Bool
        Keep only tapers with eigenvalues > 0.9
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.


    Returns
    -------
    v, e : tuple,
        v is an array of DPSS windows shaped (Kmax, N)
        e are the eigenvalues

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    from scipy.interpolate import interp1d
    Kmax = int(Kmax)
    W = float(half_nbw) / N
    nidx = np.arange(N, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % N
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, half_nbw, Kmax, low_bias=False)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            I = interp1d(x, this_d, kind=interp_kind)
            d_temp = I(np.arange(0, this_d.shape[-1] - 1,
                                 float(this_d.shape[-1] - 1) / N))

            # Rescale:
            d_temp = d_temp / np.sqrt(sum_squared(d_temp))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        # here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here I set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        diagonal = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
        # put the diagonals in LAPACK "packed" storage
        ab = np.zeros((2, N), 'd')
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest Kmax eigenvalues
        w = linalg.eigvals_banded(ab, select='i',
                                  select_range=(N - Kmax, N - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, N)
        dpss = np.zeros((Kmax, N), 'd')
        for k in range(Kmax):
            dpss[k] = tridi_inverse_iteration(diagonal, off_diag, w[k],
                                              x0=np.sin((k + 1) * t))

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    fix_skew = (dpss[1::2, 1] < 0)
    for i, f in enumerate(fix_skew):
        if f:
            dpss[2 * i + 1] *= -1

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390

    # compute autocorr using FFT (same as nitime.utils.autocorr(dpss) * N)
    rxx_size = 2 * N - 1
    n_fft = 2 ** int(np.ceil(np.log2(rxx_size)))
    dpss_fft = fftpack.fft(dpss, n_fft)
    dpss_rxx = np.real(fftpack.ifft(dpss_fft * dpss_fft.conj()))
    dpss_rxx = dpss_rxx[:, :N]

    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    if low_bias:
        idx = (eigvals > 0.9)
        if not idx.any():
            warnings.warn('Could not properly use low_bias, '
                          'keeping lowest-bias taper')
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    assert len(dpss) > 0  # should never happen
    return dpss, eigvals


def _psd_from_mt_adaptive(x_mt, eigvals, freq_mask, max_iter=150,
                          return_weights=False):
    """
    Perform an iterative procedure to compute the PSD from tapered spectra
    using the optimal weights.

    Note: Modified from NiTime

    Parameters
    ----------

    x_mt : array, shape=(n_signals, n_tapers, n_freqs)
       The DFTs of the tapered sequences (only positive frequencies)
    eigvals : array, length n_tapers
       The eigenvalues of the DPSS tapers
    freq_mask : array
        Frequency indices to keep
    max_iter : int
       Maximum number of iterations for weight computation
    return_weights : bool
       Also return the weights

    Returns
    -------
    psd : array, shape=(n_signals, np.sum(freq_mask))
        The computed PSDs
    weights : array shape=(n_signals, n_tapers, np.sum(freq_mask))
        The weights used to combine the tapered spectra

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`
    """
    n_signals, n_tapers, n_freqs = x_mt.shape

    if len(eigvals) != n_tapers:
        raise ValueError('Need one eigenvalue for each taper')

    if n_tapers < 3:
        raise ValueError('Not enough tapers to compute adaptive weights.')

    rt_eig = np.sqrt(eigvals)

    # estimate the variance from an estimate with fixed weights
    psd_est = _psd_from_mt(x_mt, rt_eig[np.newaxis, :, np.newaxis])
    x_var = np.trapz(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
    del psd_est

    # allocate space for output
    psd = np.empty((n_signals, np.sum(freq_mask)))

    # only keep the frequencies of interest
    x_mt = x_mt[:, :, freq_mask]

    if return_weights:
        weights = np.empty((n_signals, n_tapers, psd.shape[1]))

    for i, (xk, var) in enumerate(zip(x_mt, x_var)):
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = _psd_from_mt(xk[:2, :], rt_eig[:2, np.newaxis])

        err = np.zeros_like(xk)
        for n in range(max_iter):
            d_k = (psd_iter / (eigvals[:, np.newaxis] * psd_iter +
                   (1 - eigvals[:, np.newaxis]) * var))
            d_k *= rt_eig[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged
            err -= d_k
            if np.max(np.mean(err ** 2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = _psd_from_mt(xk, d_k)
            err = d_k

        if n == max_iter - 1:
            warn('Iterative multi-taper PSD computation did not converge.',
                 RuntimeWarning)

        psd[i, :] = psd_iter

        if return_weights:
            weights[i, :, :] = d_k

    if return_weights:
        return psd, weights
    else:
        return psd


def _psd_from_mt(x_mt, weights):
    """ compute PSD from tapered spectra

    Parameters
    ----------
    x_mt : array
        Tapered spectra
    weights : array
        Weights used to combine the tapered spectra

    Returns
    -------
    psd : array
        The computed PSD
    """
    psd = weights * x_mt
    psd = (psd * psd.conj()).real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd


def _csd_from_mt(x_mt, y_mt, weights_x, weights_y):
    """ Compute CSD from tapered spectra

    Parameters
    ----------
    x_mt : array
        Tapered spectra for x
    y_mt : array
        Tapered spectra for y
    weights_x : array
        Weights used to combine the tapered spectra of x_mt
    weights_y : array
        Weights used to combine the tapered spectra of y_mt

    Returns
    -------
    psd: array
        The computed PSD
    """
    csd = np.sum(weights_x * x_mt * (weights_y * y_mt).conj(), axis=-2)
    denom = (np.sqrt((weights_x * weights_x.conj()).real.sum(axis=-2)) *
             np.sqrt((weights_y * weights_y.conj()).real.sum(axis=-2)))
    csd *= 2 / denom
    return csd


def _mt_spectra(x, dpss, sfreq, n_fft=None):
    """ Compute tapered spectra

    Parameters
    ----------
    x : array, shape=(n_signals, n_times)
        Input signal
    dpss : array, shape=(n_tapers, n_times)
        The tapers
    sfreq : float
        The sampling frequency
    n_fft : int | None
        Length of the FFT. If None, the number of samples in the input signal
        will be used.

    Returns
    -------
    x_mt : array, shape=(n_signals, n_tapers, n_times)
        The tapered spectra
    freqs : array
        The frequency points in Hz of the spectra
    """

    if n_fft is None:
        n_fft = x.shape[1]

    # remove mean (do not use in-place subtraction as it may modify input x)
    x = x - np.mean(x, axis=-1)[:, np.newaxis]
    x_mt = fftpack.fft(x[:, np.newaxis, :] * dpss, n=n_fft)

    # only keep positive frequencies
    freqs = fftpack.fftfreq(n_fft, 1. / sfreq)
    freq_mask = (freqs >= 0)

    x_mt = x_mt[:, :, freq_mask]
    freqs = freqs[freq_mask]

    return x_mt, freqs


@verbose
def multitaper_psd(x, sfreq=2 * np.pi, fmin=0, fmax=np.inf, bandwidth=None,
                   adaptive=False, low_bias=True, n_jobs=1,
                   normalization='length', verbose=None):
    """Compute power spectrum density (PSD) using a multi-taper method

    Parameters
    ----------
    x : array, shape=(n_signals, n_times) or (n_times,)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    n_jobs : int
        Number of parallel jobs to use (only used if adaptive=True).
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    psd : array, shape=(n_signals, len(freqs)) or (len(freqs),)
        The computed PSD.
    freqs : array
        The frequency points in Hz of the PSD.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    if normalization not in ('length', 'full'):
        raise ValueError('Normalization must be "length" or "full", not %s'
                         % normalization)
    if x.ndim > 2:
        raise ValueError('x can only be 1d or 2d')

    x_in = np.atleast_2d(x)

    n_times = x_in.shape[1]

    # compute standardized half-bandwidth
    if bandwidth is not None:
        half_nbw = float(bandwidth) * n_times / (2 * sfreq)
    else:
        half_nbw = 4

    n_tapers_max = int(2 * half_nbw)

    dpss, eigvals = dpss_windows(n_times, half_nbw, n_tapers_max,
                                 low_bias=low_bias)

    # compute the tapered spectra
    x_mt, freqs = _mt_spectra(x_in, dpss, sfreq)

    # descide which frequencies to keep
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    # combine the tapered spectra
    if adaptive and len(eigvals) < 3:
        warn('Not adaptively combining the spectral estimators '
             'due to a low number of tapers.')
        adaptive = False

    if not adaptive:
        x_mt = x_mt[:, :, freq_mask]
        weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
        psd = _psd_from_mt(x_mt, weights)
    else:
        parallel, my_psd_from_mt_adaptive, n_jobs = \
            parallel_func(_psd_from_mt_adaptive, n_jobs)
        out = parallel(my_psd_from_mt_adaptive(x, eigvals, freq_mask)
                       for x in np.array_split(x_mt, n_jobs))
        psd = np.concatenate(out)

    if x.ndim == 1:
        # return a 1d array if input was 1d
        psd = psd[0, :]

    freqs = freqs[freq_mask]
    if normalization == 'full':
        psd /= sfreq

    return psd, freqs
