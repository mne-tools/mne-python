# Authors:  Nicolas Barascud
#           Dirk Gütlin <dirk.guetlin@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
from scipy import signal
from scipy.linalg import toeplitz
from scipy.spatial.distance import cdist, euclidean
from scipy.special import gamma, gammaincinv


def fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                         fit_quantiles=[0.022, 0.6], step_sizes=[0.01, 0.01],
                         shape_range=np.arange(1.7, 3.5, 0.15)):
    """Estimate the mean and SD of clean EEG from contaminated data.

    This function estimates the mean and standard deviation of clean EEG from
    a sample of amplitude values (that have preferably been computed over
    short windows) that may include a large fraction of contaminated samples.
    The clean EEG is assumed to represent a generalized Gaussian component in
    a mixture with near-arbitrary artifact components. By default, at least
    25% (`min_clean_fraction`) of the data must be clean EEG, and the rest
    can be contaminated. No more than 10% (`max_dropout_fraction`) of the
    data is allowed to come from contaminations that cause lower-than-EEG
    amplitudes (e.g., sensor unplugged). There are no restrictions on
    artifacts causing larger-than-EEG amplitudes, i.e., virtually anything is
    handled (with the exception of a very unlikely type of distribution that
    combines with the clean EEG samples into a larger symmetric generalized
    Gaussian peak and thereby "fools" the estimator). The default parameters
    should work for a wide range of applications but may be adapted to
    accommodate special circumstances.
    The method works by fitting a truncated generalized Gaussian whose
    parameters are constrained by `min_clean_fraction`,
    `max_dropout_fraction`, `fit_quantiles`, and `shape_range`. The fit is
    performed by a grid search that always finds a close-to-optimal solution
    if the above assumptions are fulfilled.

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        EEG data, possibly containing artifacts.
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.25).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG
        (default=0.1).
    fit_quantiles : 2-tuple
        Quantile range [lower,upper] of the truncated generalized Gaussian
        distribution that shall be fit to the EEG contents (default=[0.022
        0.6]).
    step_sizes : 2-tuple
        Step size of the grid search; the first value is the stepping of the
        lower bound (which essentially steps over any dropout samples), and
        the second value is the stepping over possible scales (i.e., clean-
        data quantiles) (default=[0.01, 0.01]).
    beta : array
        Range that the clean EEG distribution's shape parameter beta may take.

    Returns
    -------
    mu : array
        Estimated mean of the clean EEG distribution.
    sig : array
        Estimated standard deviation of the clean EEG distribution.
    alpha : float
        Estimated scale parameter of the generalized Gaussian clean EEG
        distribution.
    beta : float
        Estimated shape parameter of the generalized Gaussian clean EEG
        distribution.

    """
    # sort data so we can access quantiles directly
    X = np.sort(X)
    n = len(X)

    # compute z bounds for the truncated standard generalized Gaussian pdf and
    # pdf rescaler
    quants = np.array(fit_quantiles)
    zbounds = []
    rescale = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b], np.sign(quants - 1 / 2) * (2 * quants - 1))
        zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    # determine the quantile-dependent limits for the grid search
    # we can generally skip the tail below the lower quantile
    lower_min = np.min(quants)
    # maximum width is the fit interval if all data is clean
    max_width = np.diff(quants)
    # minimum width of the fit interval, as fraction of data
    min_width = min_clean_fraction * max_width

    # Build quantile interval matrix
    cols = np.arange(lower_min,
                     lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
                     step_sizes[0])
    cols = np.round(n * cols).astype(int)
    rows = np.arange(0, int(np.round(n * max_width)))
    newX = np.zeros((len(rows), len(cols)))
    for i, c in enumerate(range(len(rows))):
        newX[i] = X[c + cols]

    # subtract baseline value for each interval
    X1 = newX[0, :]
    newX = newX - X1

    opt_val = np.inf
    opt_lu = np.inf
    opt_bounds = np.inf
    opt_beta = np.inf
    gridsearch = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
    for m in gridsearch.astype(int):
        mcurr = m - 1
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        cols = nbins / newX[mcurr]
        H = newX[:m] * cols

        hist_all = []
        for ih in range(len(cols)):
            histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
            hist_all.append(histcurr[0])
        hist_all = np.array(hist_all, dtype=int).T
        hist_all = np.vstack((hist_all, np.zeros(len(cols), dtype=int)))
        logq = np.log(hist_all + 0.01)

        # for each shape value...
        for k, b in enumerate(shape_range):
            bounds = zbounds[k]
            x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)  # noqa:E501
            p = np.exp(-np.abs(x) ** b) * rescale[k]
            p = p / np.sum(p)

            # calc KL divergences
            kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)

            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = shape_range[k]
                opt_bounds = bounds
                opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


def yulewalk(order, F, M):
    """Recursive filter design using a least-squares method.

    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:
    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)
    matches the magnitude frequency response given by vectors F and M.
    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.
    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed
    explanation of the algorithm see [1]_.

    Parameters
    ----------
    order : int
        Filter order.
    F : array
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : array
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.

    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace
           Electronic Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.

    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:
    >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    >>> [b, a] = yulewalk(8, f, m) # Filter design using least-squares method

    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=None)
    A = polystab(denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=np.complex))))  # noqa
    )
    B = np.real(numf(hh[0:nr], A, nb))

    return B, A


def yulewalk_filter(X, sfreq, zi=None, ab=None, axis=-1):
    """Yulewalk filter.

    Parameters
    ----------
    X : array, shape = (n_channels, n_samples)
        Data to filter.
    sfreq : float
        Sampling frequency.
    zi : array, shape=(n_channels, filter_order)
        Initial conditions.
    a, b : 2-tuple | None
        Coefficients of an IIR filter that is used to shape the spectrum of
        the signal when calculating artifact statistics. The output signal
        does not go through this filter. This is an optional way to tune the
        sensitivity of the algorithm to each frequency component of the
        signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies.
    axis : int
        Axis to filter on (default=-1, corresponding to samples).

    Returns
    -------
    out : array
        Filtered data.
    zf :  array, shape=(n_channels, filter_order)
        Output filter state.
    """
    # Set default IIR filter coefficients
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (sfreq / 2.0) - 1.0), sfreq / 2.0]) * 2.0 / sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab

    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        out = signal.lfilter(B, A, X, axis=axis)
        zf = None
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf


def ma_filter(N, X, Zi):
    """Run a moving average filter over the data.

    Parameters
    ----------
    N : int
        Length of the filter.
    X : array, shape=(n_channels, n_samples)
        The raw data.
    Zi : array
        The initial filter conditions.

    Returns
    -------
    X : array
        The filtered data.
    Zf : array
        The new fiter conditions.
    """
    if Zi is None:
        Zi = np.zeros([len(X), N])

    Y = np.concatenate([Zi, X], axis=1)
    M = Y.shape[-1]
    I_ = np.stack([np.arange(M - N),
                   np.arange(N, M)]).astype(int)
    S = (np.stack([-np.ones(M - N),
                   np.ones(M - N)]) / N)
    X = np.cumsum(np.multiply(Y[:, np.reshape(I_.T, -1)],
                              np.reshape(S.T, [-1])), axis=-1)

    X = X[:, 1::2]

    Zf = np.concatenate([-(X[:, -1] * N - Y[:, -N])[:, np.newaxis],
                         Y[:, -N + 1:]], axis=-1)
    return X, Zf


def geometric_median(X, tol=1e-5, max_iter=500):
    """Geometric median.

    This code is adapted from [2]_ using the Vardi and Zhang algorithm
    described in [1]_.

    Parameters
    ----------
    X : array, shape=(n_observations, n_variables)
        The data.
    tol : float
        Tolerance (default=1.e-5)
    max_iter : int
        Max number of iterations (default=500):

    Returns
    -------
    y1 : array, shape=(n_variables,)
        Geometric median over X.

    References
    ----------
    .. [1] Vardi, Y., & Zhang, C. H. (2000). The multivariate L1-median and
       associated data depth. Proceedings of the National Academy of Sciences,
       97(4), 1423-1426. https://doi.org/10.1073/pnas.97.4.1423
    .. [2] https://stackoverflow.com/questions/30299267/

    """
    y = np.mean(X, 0)  # initial value

    i = 0
    while i < max_iter:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1. / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < tol:
            return y1

        y = y1
        i += 1
    else:
        print(f"Geometric median could converge in {i} iterations "
              f"with a tolerance of {tol}")


def polystab(a):
    """Polynomial stabilization.

    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.

    Parameters
    ----------
    a : array
        The vector of polynomial coefficients.

    Returns
    -------
    b : array
        The stabilized polynomial.

    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.
    >>> h = fir1(25,0.4);               # Window-based FIR filter design
    >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    >>> flag_minphase = isminphase(hmin)# Determines if filter is min phase

    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not(np.sum(np.imag(a))):
        b = np.real(b)

    return b


def numf(h, a, nb):
    """Get numerator B given impulse-response h of B/A and denominator A."""
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)

    b = np.linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b


def denf(R, na):
    """Compute order NA denominator A from covariances R(0)...R(nr)."""
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, np.linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A


def block_covariance(data, window=128):
    """Compute blockwise covariance.

    Parameters
    ----------
    data : array, shape=(n_chans, n_samples)
        Input data (must be 2D)
    window : int
        Window size.

    Returns
    -------
    cov : array, shape=(n_blocks, n_chans, n_chans)
        Block covariance.
    """
    n_ch, n_times = data.shape
    U = np.zeros([len(np.arange(0, n_times - 1, window)), n_ch**2])
    data = data.T
    for k in range(0, window):
        idx_range = np.minimum(n_times - 1,
                               np.arange(k, n_times + k - 2, window))
        U = U + np.reshape(data[idx_range].reshape([-1, 1, n_ch]) *
                           data[idx_range].reshape(-1, n_ch, 1), U.shape)

    return np.array(U)
