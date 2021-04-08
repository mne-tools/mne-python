"""Artifact Subspace Reconstruction."""
# TODO: uninstall statsmodels, pymanopt after testing equivalence
import logging

import numpy as np
from scipy import linalg, signal
from scipy.stats import median_abs_deviation

from .asr_utils import (sliding_window, geometric_median, fit_eeg_distribution, yulewalk,
                        yulewalk_filter, block_covariance, nonlinear_eigenspace)

try:
    import pyriemann
except ImportError:
    pyriemann = None
try:
    import pymanopt
except ImportError:
    pymanopt = None


class ASR():
    """Artifact Subspace Reconstruction.

    Artifact subspace reconstruction (ASR) is an automatic, online,
    component-based artifact removal method for removing transient or
    large-amplitude artifacts in multi-channel EEG recordings [1]_.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data, in Hz.

    The following are optional parameters (the key parameter of the method is
    the ``cutoff``):

    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to Channels x Channels x Samples
        x 16 / Blocksize bytes) (default=10).
    win_len : float
        Window length (s) that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts but
        not shorter than half a cycle of the high-pass filter that was used
        (default=1).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'riemann', 'euclid'}
        Method to use. If riemann, use the riemannian-modified version of
        ASR [2]_.
    memory : float
        Memory size (s), regulates the number of covariance matrices to store.
    estimator : str in {'scm', 'lwf', 'oas', 'mcd'}
        Covariance estimator (default: 'scm' which computes the sample
        covariance). Use 'lwf' if you need regularization (requires pyriemann).

    Attributes
    ----------
    ``state_`` : dict
        Initial state of the ASR filter.
    ``zi_``: array, shape=(n_channels, filter_order)
        Filter initial conditions.
    ``ab_``: 2-tuple
        Coefficients of an IIR filter that is used to shape the spectrum of the
        signal when calculating artifact statistics. The output signal does not
        go through this filter. This is an optional way to tune the sensitivity
        of the algorithm to each frequency component of the signal. The default
        filter is less sensitive at alpha and beta frequencies and more
        sensitive at delta (blinks) and gamma (muscle) frequencies.
    ``cov_`` : array, shape=(channels, channels)
        Previous covariance matrix.
    ``state_`` : dict
        Previous ASR parameters (as derived by :func:`asr_calibrate`) for
        successive calls to :meth:`transform`. Required fields are:

        - ``M`` : Mixing matrix
        - ``T`` : Threshold matrix
        - ``R`` : Reconstruction matrix (array | None)

    References
    ----------
    .. [1] Kothe, C. A. E., & Jung, T. P. (2016). U.S. Patent Application No.
       14/895,440. https://patents.google.com/patent/US20160113587A1/en
    .. [2] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S.
       (2019). A Riemannian Modification of Artifact Subspace Reconstruction
       for EEG Artifact Handling. Frontiers in Human Neuroscience, 13.
       https://doi.org/10.3389/fnhum.2019.00141

    """

    def __init__(self, sfreq=250, cutoff=5, blocksize=100, win_len=0.5,
                 win_overlap=0.66, max_dropout_fraction=0.1,
                 min_clean_fraction=0.25, **kwargs):

        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = 0.3
        self.memory = int(2 * sfreq)  # smoothing window for covariances
        self.sample_weight = np.geomspace(0.05, 1, num=self.memory + 1)
        self.sfreq = sfreq

        self.reset()

    def reset(self):
        """Reset filter."""
        # Initialise yulewalk-filter coefficients with sensible defaults
        F = np.array([0, 2, 3, 13, 16, 40,
                      np.minimum(80.0, (self.sfreq / 2.0) - 1.0),
                      self.sfreq / 2.0]) * 2.0 / self.sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
        self.ab_ = (A, B)
        self.cov_ = []
        self.zi_ = None
        self.state_ = {}
        self._counter = []
        self._fitted = False

    def fit(self, X, y=None, **kwargs):
        """Calibration for the Artifact Subspace Reconstruction method.

        The input to this data is a multi-channel time series of calibration
        data. In typical uses the calibration data is clean resting EEG data of
        data if the fraction of artifact content is below the breakdown point
        of the robust statistics used for estimation (50% theoretical, ~30%
        practical). If the data has a proportion of more than 30-50% artifacts
        then bad time windows should be removed beforehand. This data is used
        to estimate the thresholds that are used by the ASR processing function
        to identify and remove artifact components.

        The calibration data must have been recorded for the same cap design
        from which data for cleanup will be recorded, and ideally should be
        from the same session and same subject, but it is possible to reuse the
        calibration data from a previous session and montage to the extent that
        the cap is placed in the same location (where loss in accuracy is more
        or less proportional to the mismatch in cap placement).

        Parameters
        ----------
        X : array, shape=(n_channels, n_samples)
            The calibration data should have been high-pass filtered (for
            example at 0.5Hz or 1Hz using a Butterworth IIR filter), and be
            reasonably clean not less than 30 seconds (this method is typically
            used with 1 minute or more).

        """
        if X.ndim == 3:
            X = X.squeeze()

        # Find artifact-free windows first
        clean, sample_mask = clean_windows(
            X,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction)


        # Perform calibration
        M, T = asr_calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            method=self.method,
            estimator=self.estimator)

        self.state_ = dict(M=M, T=T, R=None)
        self._fitted = True

        return clean, sample_mask

    def transform(self, X, y=None, **kwargs):
        """Apply Artifact Subspace Reconstruction.

        Parameters
        ----------
        X : array, shape=([n_trials, ]n_channels, n_samples)
            Raw data.

        Returns
        -------
        out : array, shape=([n_trials, ]n_channels, n_samples)
            Filtered data.

        """
        if X.ndim == 3:
            if X.shape[0] == 1:  # single epoch case
                out = self.transform(X[0])
                return out[None, ...]
            else:
                outs = [self.transform(x) for x in X]
                return np.stack(outs, axis=0)
        else:
            # Yulewalk-filtered data
            X_filt, self.zi_ = yulewalk_filter(
                X, sfreq=self.sfreq, ab=self.ab_, zi=self.zi_)

        if not self._fitted:
            logging.warning('ASR is not fitted ! Returning unfiltered data.')
            return X

        if self.estimator == 'scm':
            cov = 1 / X.shape[-1] * X_filt @ X_filt.T
        else:
            cov = pyriemann.estimation.covariances(X_filt[None, ...],
                                                   self.estimator)[0]

        self._counter.append(X_filt.shape[-1])
        self.cov_.append(cov)

        # Regulate the number of covariance matrices that are stored
        while np.sum(self._counter) > self.memory:
            if len(self.cov_) > 1:
                self.cov_.pop(0)
                self._counter.pop(0)
            else:
                self._counter = [self.memory, ]
                break

        # Exponential covariance weights – the most recent covariance has a
        # weight of 1, while the oldest one in memory has a weight of 5%
        weights = [1, ]
        for c in np.cumsum(self._counter[1:]):
            weights = [self.sample_weight[-c]] + weights

        # Clean data, using covariances weighted by sample_weight
        out, self.state_ = asr_process(X, X_filt, self.state_,
                                       cov=np.stack(self.cov_),
                                       method=self.method,
                                       sample_weight=weights)

        return out


def asr_calibrate(X, sfreq, cutoff=5, blocksize=100, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, method='euclid', estimator='scm'):
    """Calibration function for the Artifact Subspace Reconstruction method.

    The input to this data is a multi-channel time series of calibration data.
    In typical uses the calibration data is clean resting EEG data of ca. 1
    minute duration (can also be longer). One can also use on-task data if the
    fraction of artifact content is below the breakdown point of the robust
    statistics used for estimation (50% theoretical, ~30% practical). If the
    data has a proportion of more than 30-50% artifacts then bad time windows
    should be removed beforehand. This data is used to estimate the thresholds
    that are used by the ASR processing function to identify and remove
    artifact components.

    The calibration data must have been recorded for the same cap design from
    which data for cleanup will be recorded, and ideally should be from the
    same session and same subject, but it is possible to reuse the calibration
    data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less
    proportional to the mismatch in cap placement).

    The calibration data should have been high-pass filtered (for example at
    0.5Hz or 1Hz using a Butterworth IIR filter).

    Parameters
    ----------
    X : array, shape=([n_trials, ]n_channels, n_samples)
        *zero-mean* (e.g., high-pass filtered) and reasonably clean EEG of not
        much less than 30 seconds (this method is typically used with 1 minute
        or more).
    sfreq : float
        Sampling rate of the data, in Hz.

    The following are optional parameters (the key parameter of the method is
    the ``cutoff``):

    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to n_chans x n_chans x n_samples
        x 16 / blocksize bytes) (default=100).
    win_len : float
        Window length that is used to check the data for artifact content. This
        is ideally as long as the expected time scale of the artifacts but
        short enough to allow for several 1000 windows to compute statistics
        over (default=0.5).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average.

    Returns
    -------
    M : array
        Mixing matrix.
    T : array
        Threshold matrix.

    """
    logging.debug('[ASR] Calibrating...')

    [nc, ns] = X.shape

    X, _zf = yulewalk_filter(X, 128)  # TODO: REPLACE WITH ACTUAL SAMP RATE?

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    U = block_covariance(X, window=blocksize, overlap=win_overlap,
                         estimator=estimator)
    if method == 'euclid':
        Uavg = geometric_median(U.reshape((-1, nc * nc))/blocksize)
        Uavg = Uavg.reshape((nc, nc))
    else:  # method == 'riemann'
        Uavg = pyriemann.utils.mean_covariance(U, metric='riemann')

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))
    #D, Vtmp = linalg.eigh(M)
    D, Vtmp = nonlinear_eigenspace(M, nc) if method == "riemann" else linalg.eigh(M)
    V = -Vtmp[:, np.argsort(D)]

    # get the threshold matrix T
    x = np.abs(np.dot(V.T, X))
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))

    mu = np.zeros(nc)
    sig = np.zeros(nc)

    for ichan in reversed(range(nc)):
        rms = x[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(rms[o:o + N]) / N))
        mu[ichan], sig[ichan], alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction)
    T = np.dot(np.diag(mu + cutoff * sig), V.T)

    logging.debug('[ASR] Calibration done.')
    return M, T


def asr_processs(data, srate, state, windowlen, lookahead, stepsize, maxdims):

    if maxdims < 1:
        maxdims = np.round(len(data)*maxdims)

    C, S = data.shape

    N = np.round(windowlen*srate).astype(int)
    P = np.round(lookahead*srate).astype(int)

    T,M,A,B = state.T, state.M, state.A, state.B

    if state.carry == None:
        state.carry = np.tile(2*data[:,0], (P, 1)).T - data[:, np.mod(np.arange(P,0,-1),S)]
    data = np.concatenate([state.carry, data], axis=-1)
    # matlab does replace infs (/nans?) witzh 0 here

    #splits = np.ceil(C*C*S*8*8 + C*C*8*s/stepsize + C*S*8*2 + S*8*5)
    splits=3  # TODO: use this for parallelization

    for i in range(splits):

        i_range = np.arange(i*S//splits, np.min([(i+1)*S//splits, S]), dtype=int)
        X, state.iir = yulewalk_filter(data[:, i_range + P], srate, zi=state.iir, ab=(A, B), axis=-1)

        Xcov, state.cov = moving_average(N, np.reshape(np.multiply(np.reshape(X, (1, C, -1)), np.reshape(X, (C, 1, -1))), (C*C, -1)), state.cov)

        update_at = np.minimum(np.arange(stepsize, Xcov.shape[-1]+stepsize-2, stepsize), Xcov.shape[-1])-1

        if state.last_R is None:
            update_at = np.concatenate([[0], update_at])
            state.last_R = np.eye(C)

        Xcov = np.reshape(Xcov[:, update_at], (C, C, -1))

        last_n = 0
        for j in range(len(update_at)-1):
            D, V = np.linalg.eigh(Xcov[:, :, j]) # TODO: revert
            #V = np.multiply(np.arange(C)+1, np.arange(C)[:, np.newaxis]+1)
            #D = np.arange(C, 0, -1) * 5000000000000

            keep = np.logical_or(D < np.sum((T@V)**2, axis=0), np.arange(C)+1 < (C-maxdims))

            trivial = np.all(keep)
            if not trivial:
                from scipy.sparse.linalg import lsqr
                #print(np.multiply(keep[:, np.newaxis], V.T @ M)[:, -1])
                #print(M[:3, :3], M[-3:, -3:], V[:3, :3], V[-3:, -3:])
                #R = np.real(M @ np.linalg.pinv(np.multiply(keep[:, np.newaxis], V.T @ M)) @ V.T) # TODO: CHANGE BACK
                R = np.real(M @ np.linalg.pinv(np.multiply(keep[:, np.newaxis], V.T @ M)) @ V.T)
                #R = np.real(M @ (np.multiply(keep[:, np.newaxis], V.T @ M)).T @ V.T)
                print("R AFTER INIT ITer: {0} {1}".format(i, j), R[:3, :3], R.shape)
            else:
                R = np.eye(C)

            # TODO: i think there's one iteration missing compared to matlab
            n = update_at[j] + 1
            if (not trivial) or (not state.last_trivial):
                print(n)
                subrange = i_range[np.arange(last_n, n)]

                blend = (1 - np.cos(np.pi * np.arange(1, n-last_n+1)/(n-last_n)))/2


                data[:, subrange] = np.multiply(blend, R@data[:, subrange]) + np.multiply(1-blend, state.last_R@data[:, subrange])
                print("SUBRANGE: ", subrange)
                print("data ITer: {0} {1}".format(i, j), data[:3, subrange], data[:, subrange].shape)
            last_n, state.last_R, state.last_trivial = n, R, trivial

            #if j > 20:
            #    raise ValueError

    state.carry = np.concatenate([state.carry, data[:, -P:]])
    state.carry = state.carry[:, -P:]

    return data[:, :-P], state


def asr_process(data, sfreq, M, T, windowlen=0.5, lookahead=0.25, stepsize=32, maxdims=0.66, A=None, B=None, R=None, Zi=None, cov=None, carry=None, return_states=False):
    """Apply Artifact Subspace Reconstruction method.

    This function is used to clean multi-channel signal using the ASR method.
    The required inputs are the data matrix, the sampling rate of the data, and
    the filter state.

    Parameters
    ----------
    data : array, shape=([n_trials, ]n_channels, n_samples)
        Raw data.
    M : array, shape=(n_channels, n_channels)
        Mixing matrix.
    T : array, shape=(n_channels, n_channels)
        Threshold matrix.
    max_dims : float, int
        Maximum dimensionality of artifacts to remove. This parameter
        denotes the maximum number of dimensions which can be removed from
        each segment. If larger than 1, `int(max_dims)` will denote the
        maximum number of dimensions removed from the data. If smaller than 1,
        `max_dims` describes a fraction of total dimensions. Defaults to 0.66.
    T : array, shape=(n_channels, n_channels)
        Previous reconstruction matrix.
    cov : array, shape=([n_trials, ]n_channels, n_channels) | None
        Covariance. If None (default), then it is computed from ``X_filt``. If
        a 3D array is provided, the average covariance is computed from all the
        elements in it.


    Returns
    -------
    clean : array, shape=([n_trials, ]n_channels, n_samples)
        Clean data.
    state : dict
        Output ASR parameters {"M":M, "T":T, "R":R, "Zi":Zi, "cov":cov, "carry":carry}.

    """

    if maxdims < 1:
        maxdims = np.round(len(data)*maxdims)

    if Zi is None:
        Zi = yulewalk_filter(data, ab=(A, B), sfreq=sfreq, zi=np.ones([len(data), 8]))

    C, S = data.shape

    N = np.round(windowlen*sfreq).astype(int)
    P = np.round(lookahead*sfreq).astype(int)

    if carry == None:
        carry = np.tile(2*data[:,0], (P, 1)).T - data[:, np.mod(np.arange(P,0,-1),S)]
    data = np.concatenate([carry, data], axis=-1)

    # matlab does replace infs (/nans?) witzh 0 here

    #splits = np.ceil(C*C*S*8*8 + C*C*8*s/stepsize + C*S*8*2 + S*8*5)...
    splits=3  # TODO: use this for parallelization

    last_trivial = False
    last_R = None
    for i in range(splits):

        i_range = np.arange(i*S//splits, np.min([(i+1)*S//splits, S]), dtype=int)
        X, Zi = yulewalk_filter(data[:, i_range + P], sfreq, zi=Zi, ab=(A, B), axis=-1)

        Xcov, cov = moving_average(N, np.reshape(np.multiply(np.reshape(X, (1, C, -1)), np.reshape(X, (C, 1, -1))), (C*C, -1)), cov)

        update_at = np.minimum(np.arange(stepsize, Xcov.shape[-1]+stepsize-2, stepsize), Xcov.shape[-1])-1

        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = np.eye(C)

        Xcov = np.reshape(Xcov[:, update_at], (C, C, -1))

        last_n = 0
        for j in range(len(update_at)-1):
            D, V = np.linalg.eigh(Xcov[:, :, j])

            keep = np.logical_or(D < np.sum((T@V)**2, axis=0), np.arange(C)+1 < (C-maxdims))
            trivial = np.all(keep)

            if not trivial:
                R = np.real(M @ np.linalg.pinv(np.multiply(keep[:, np.newaxis], V.T @ M)) @ V.T)
            else:
                R = np.eye(C)

            # TODO: there might be one iteration missing compared to matlab
            n = update_at[j] + 1
            if (not trivial) or (not last_trivial):

                subrange = i_range[np.arange(last_n, n)]
                blend = (1 - np.cos(np.pi * np.arange(1, n-last_n+1)/(n-last_n)))/2

                data[:, subrange] = np.multiply(blend, R@data[:, subrange]) + np.multiply(1-blend, last_R@data[:, subrange])
            last_n, last_R, last_trivial = n, R, trivial

    carry = np.concatenate([carry, data[:, -P:]])
    carry = carry[:, -P:]

    if return_states:
        return data[:, :-P], {"M":M, "T":T, "R":R, "Zi":Zi, "cov":cov, "carry":carry}
    else:
        return data[:, :-P]



def moving_average(N, X, Zi):
    if Zi is None:
        Zi = np.zeros([len(X), N])

    Y = np.concatenate([Zi, X], axis=1)
    M = Y.shape[-1]
    I = np.stack([np.arange(M-1-(N-1)),
                  np.arange(N, M)]).astype(int)
    S = (np.stack([-np.ones(M-N),
                   np.ones(M-N)])/N)
    X = np.cumsum(np.multiply(Y[:, np.reshape(I.T, -1)], np.reshape(S.T, [-1])), axis=-1)

    X = X[:, 1::2]

    Zf = np.concatenate([-(X[:,-1] * N - Y[:, -N])[:, np.newaxis],
                     Y[:, -N+1:]], axis=-1)
    return X, Zf



def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1, show=False):
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power
    artifacts. Specifically, only windows are retained which have less than a
    certain fraction of "bad" channels, where a channel is bad in a window if
    its power is above or below a given upper/lower threshold (in standard
    deviations from a robust estimate of the EEG power distribution in the
    channel).

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        Continuous data set, assumed to be appropriately high-passed (e.g. >
        1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    zthresholds : 2-tuple
        The minimum and maximum standard deviations within which the power of a
        channel must lie (relative to a robust estimate of the clean EEG power
        distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).

    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.

    win_len : float
        Window length that is used to check the data for artifact content. This
        is ideally as long as the expected time scale of the artifacts but not
        shorter than half a cycle of the high-pass filter that was used.
        Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)

    The following are expert-level parameters that you should not tune unless
    you fully understand how the method works.

    truncate_quant :
        Truncated Gaussian quantile. Quantile range [upper,lower] of the
        truncated Gaussian distribution that shall be fit to the EEG contents.
        (default=[0.022, 0.6])
    step_sizes :
        Grid search stepping. Step size of the grid search, in quantiles;
        separately for [lower,upper] edge of the truncated Gaussian. The lower
        edge has finer stepping because the clean data density is assumed to be
        lower there, so small changes in quantile amount to large changes in
        data space (default=[0.01 0.01]).
    shape_range :
        Shape parameter range. Search range for the shape parameter of the
        generalized Gaussian distribution used to fit clean EEG (default:
        1.7:0.15:3.5).

    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).

    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.linspace(1.7, 3.5, 13)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):
        x = X[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(x[o:o + N]) / N))

        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(np.int(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + np.int(max_bad_chans - 1), :] < np.min(zthresholds))

    bad_by_mad = median_abs_deviation(wz, scale=1, axis=0) < .1
    bad_by_std = np.std(wz, axis=0) < .1
    mask3 = np.logical_or(bad_by_mad, bad_by_std)

    remove_mask = np.logical_or.reduce((mask1, mask2, mask3))
    removed_wins = np.where(remove_mask)

    sample_maskidx = []
    for i in range(len(removed_wins[0])):
        if i == 0:
            sample_maskidx = np.arange(
                offsets[removed_wins[0][i]], offsets[removed_wins[0][i]] + N)
        else:
            sample_maskidx = np.vstack((
                sample_maskidx,
                np.arange(offsets[removed_wins[0][i]],
                          offsets[removed_wins[0][i]] + N)
            ))

    sample_mask2remove = np.unique(sample_maskidx)
    clean = np.delete(X, sample_mask2remove, 1) if sample_mask2remove.size else X
    sample_mask = np.ones((1, ns), dtype=bool)

    if sample_mask2remove.size:
        sample_mask[0, sample_mask2remove] = False

    if show:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(nc, sharex=True, figsize=(8, 5))
        times = np.arange(ns) / float(sfreq)
        for i in range(nc):
            ax[i].fill_between(times, 0, 1, where=sample_mask.flat,
                               transform=ax[i].get_xaxis_transform(),
                               facecolor='none', hatch='...', edgecolor='k',
                               label='selected window')
            ax[i].plot(times, X[i], lw=.5, label='EEG')
            ax[i].set_ylim([-50, 50])
            # ax[i].set_ylabel(raw.ch_names[i])
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel(f'ch{i}')
        ax[0].legend(fontsize='small', bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle('Clean windows')
        plt.show()

    return clean, sample_mask



def qr_pinv(A):
    """Create the pseudoinverse using QR solver"""
    from scipy.linalg import qr, inv
    Q, R = qr(A)
    rank = np.linalg.matrix_rank(A)
    R_inv = inv(R[:rank, :rank])
    R_inv_O = np.zeros_like(A.T, dtype=float)
    R_inv_O[:rank, :rank] = R_inv

    return R_inv_O @ Q.T
