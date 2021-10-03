# Authors:  Dirk Gütlin <dirk.guetlin@gmail.com>
#           Nicolas Barascud
#
# License: BSD (3-clause)
import logging
import warnings

import numpy as np
from scipy import linalg
from numpy.linalg import pinv

from .asr_utils import (geometric_median, fit_eeg_distribution, yulewalk,
                        yulewalk_filter, ma_filter, block_covariance)


class ASR():
    """Artifact Subspace Reconstruction.

    Artifact subspace reconstruction (ASR) is an automated, online,
    component-based artifact removal method for removing transient or
    large-amplitude artifacts in multi-channel EEG recordings [1]_.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data, in Hz.
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
        x 16 / Blocksize bytes) (default=100).
    win_len : float
        Window length (s) that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts but
        not shorter than half a cycle of the high-pass filter that was used
        (default=0.5).
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
    ab : 2-tuple | None
        Coefficients (A, B) of an IIR filter that is used to shape the
        spectrum of the signal when calculating artifact statistics. The
        output signal does not go through this filter. This is an optional way
        to tune the sensitivity of the algorithm to each frequency component
        of the signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies. Defaults to None.
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    method : {'riemann', 'euclid'}
        Method to use. If riemann, use the riemannian-modified version of
        ASR [2]_. Currently, only euclidean ASR is supported. Defaults to
        "euclid".

    Attributes
    ----------
    sfreq: array, shape=(n_channels, filter_order)
        Filter initial conditions.
    cutoff: float
        Standard deviation cutoff for rejection.
    blocksize : int
        Block size for calculating the robust data covariance and thresholds.
    win_len : float
        Window length (s) that is used to check the data for artifact content.
    win_overlap : float
        Window overlap fraction.
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts.
    min_clean_fraction : float
        Minimum fraction of windows.
    max_bad_chans : float
        The maximum fraction of bad channels.
    method : {'riemann', 'euclid'}
        Method to use.
    A, B: arrays
        Coefficients of an IIR filter that is used to shape the spectrum of the
        signal when calculating artifact statistics. The output signal does not
        go through this filter. This is an optional way to tune the sensitivity
        of the algorithm to each frequency component of the signal. The default
        filter is less sensitive at alpha and beta frequencies and more
        sensitive at delta (blinks) and gamma (muscle) frequencies.
    M : array, shape=(channels, channels)
        The mixing matrix to fit ASR data.
    T : array, shape=(channels, channels)
        The mixing matrix to fit ASR data.

    References
    ----------
    .. [1] Kothe, C. A. E., & Jung, T. P. (2016). U.S. Patent Application No.
       14/895,440. https://patents.google.com/patent/US20160113587A1/en
    .. [2] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S.
       (2019). A Riemannian Modification of Artifact Subspace Reconstruction
       for EEG Artifact Handling. Frontiers in Human Neuroscience, 13.
       https://doi.org/10.3389/fnhum.2019.00141

    """

    def __init__(self, sfreq=1000, cutoff=5, blocksize=100, win_len=0.5,
                 win_overlap=0.66, max_dropout_fraction=0.1,
                 min_clean_fraction=0.25, ab=None, max_bad_chans=0.1,
                 method="euclid"):

        # set attributes
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = max_bad_chans
        self.method = "euclid"  # NOTE: riemann is not yet available
        self._fitted = False

        # set default yule-walker filter
        if ab is None:
            yw_f = np.array([0, 2, 3, 13, 16, 40,
                             np.minimum(80.0, (self.sfreq / 2.0) - 1.0),
                             self.sfreq / 2.0]) * 2.0 / self.sfreq
            yw_m = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
            self.B, self.A = yulewalk(8, yw_f, yw_m)
        else:
            self.A, self.B = ab

        self.reset()

    def reset(self):
        """Reset state variables."""
        self.M = None
        self.T = None

        # TODO: The following parameters are effectively not used. Still,
        #  they can be set manually via asr.transform(return_states=True)
        self.R = None
        self.carry = None
        self.Zi = None
        self.cov = None
        self._fitted = False

    def fit(self, X, y=None, return_clean_window=False):
        """Calibration for the Artifact Subspace Reconstruction method.

        The input to this data is a multi-channel time series of calibration
        data. In typical uses the calibration data is clean resting EEG data
        of data if the fraction of artifact content is below the breakdown
        point of the robust statistics used for estimation (50% theoretical,
        ~30% practical). If the data has a proportion of more than 30-50%
        artifacts then bad time windows should be removed beforehand. This
        data is used to estimate the thresholds that are used by the ASR
        processing function to identify and remove artifact components.

        The calibration data must have been recorded for the same cap design
        from which data for cleanup will be recorded, and ideally should be
        from the same session and same subject, but it is possible to reuse
        the calibration data from a previous session and montage to the
        extent that the cap is placed in the same location (where loss in
        accuracy is more or less proportional to the mismatch in cap
        placement).

        Parameters
        ----------
        X : array, shape=(n_channels, n_samples)
            The calibration data should have been high-pass filtered (for
            example at 0.5Hz or 1Hz using a Butterworth IIR filter), and be
            reasonably clean not less than 30 seconds (this method is
            typically used with 1 minute or more).
        y : None
            Redundant argument. Included for scikit-learn compatibility.
            Changing this argument have no effect.
        return_clean_window : Bool
            If True, the method will return the variables `clean` (the cropped
             dataset which was used to fit the ASR) and `sample_mask` (a
             logical mask of which samples were included/excluded from fitting
             the ASR). Defaults to False.

        Returns
        -------
        clean : array, shape=(n_channels, n_samples)
            The cropped version of the dataset which was used to calibrate
            the ASR. This array is a result of the `clean_windows` function
            and no ASR was applied to it.
        sample_mask : boolean array, shape=(1, n_samples)
            Logical mask of the samples which were used to train the ASR.

        """
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
        self.M, self.T = asr_calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            ab=(self.A, self.B),
            method=self.method)

        self._fitted = True

        # return data if required
        if return_clean_window:
            return clean, sample_mask

    def transform(self, X, y=None, lookahead=0.25, stepsize=32, maxdims=0.66,
                  return_states=False, mem_splits=3):
        """Apply Artifact Subspace Reconstruction.

        Parameters
        ----------
        X : array, shape=([n_trials, ]n_channels, n_samples)
            Raw data.
        y : None
            Redundant argument. Included for scikit-learn compatibility.
            Changing this argument have no effect.
        lookahead:
            Amount of look-ahead that the algorithm should use. Since the
            processing is causal, the output signal will be delayed by this
            amount. This value is in seconds and should be between 0 (no
            lookahead) and WindowLength/2 (optimal lookahead). The recommended
            value is WindowLength/2. Default: 0.25
        stepsize:
            The steps in which the algorithm will be updated. The larger this
            is, the faster the algorithm will be. The value must not be larger
            than WindowLength * SamplingRate. The minimum value is 1 (update
            for every sample) while a good value would be sfreq//3. Note that
            an update is always performed also on the first and last sample of
            the data chunk. Default: 32
        max_dims : float, int
            Maximum dimensionality of artifacts to remove. This parameter
            denotes the maximum number of dimensions which can be removed from
            each segment. If larger than 1, `int(max_dims)` will denote the
            maximum number of dimensions removed from the data. If smaller
            than 1, `max_dims` describes a fraction of total dimensions.
            Defaults to 0.66.
        return_states : bool
            If True, returns a dict including the updated states {"M":M,
            "T":T, "R":R, "Zi":Zi, "cov":cov, "carry":carry}. Defaults to
            False.
        mem_splits : int
            Split the array in `mem_splits` segments to save memory.
        
        Returns
        -------
        out : array, shape=(n_channels, n_samples)
            Filtered data.

        """
        return asr_process(X, self.sfreq, self.M, self.T, self.win_len,
                           lookahead, stepsize, maxdims, (self.A, self.B),
                           self.R, self.Zi, self.cov, self.carry,
                           return_states, self.method, mem_splits)


def asr_calibrate(X, sfreq, cutoff=5, blocksize=100, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, ab=None, method='euclid'):
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
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to n_chans x n_chans x
        n_samples x 16 / blocksize bytes) (default=100).
    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but short enough to allow for several 1000 windows to compute
        statistics over (default=0.5).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average. For now, only
        euclidean ASR is supported.

    Returns
    -------
    M : array
        Mixing matrix.
    T : array
        Threshold matrix.

    """
    if method == "riemann":
        warnings.warn("Riemannian ASR is not yet supported. Switching back to"
                      " Euclidean ASR.")
        method == "euclid"

    logging.debug('[ASR] Calibrating...')

    # set number of channels and number of samples
    [nc, ns] = X.shape

    # filter the data
    X, _zf = yulewalk_filter(X, sfreq, ab=ab)

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    # get block covariances
    U = block_covariance(X, window=blocksize)

    # get geometric median for each block
    # Note: riemann mode is not yet supported, else this could be:
    # Uavg = pyriemann.utils.mean_covariance(U, metric='riemann')
    Uavg = geometric_median(U.reshape((-1, nc * nc)) / blocksize)
    Uavg = Uavg.reshape((nc, nc))

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))

    # sort the get the sorted eigenvecotors/eigenvalues
    # riemann is not yet supported, else this could be PGA/nonlinear eigenvs
    D, Vtmp = linalg.eigh(M)
    V = Vtmp[:, np.argsort(D)]  # I think numpy sorts them automatically

    # get the threshold matrix T
    x = np.abs(np.dot(V.T, X))
    offsets = np.int_(np.arange(0, ns - N, np.round(N * (1 - win_overlap))))

    # go through all the channels and fit the EEG distribution
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


def asr_process(data, sfreq, M, T, windowlen=0.5, lookahead=0.25, stepsize=32,
                maxdims=0.66, ab=None, R=None, Zi=None, cov=None, carry=None,
                return_states=False, method="euclid", mem_splits=3):
    """Apply Artifact Subspace Reconstruction method.

    This function is used to clean multi-channel signal using the ASR method.
    The required inputs are the data matrix and the sampling rate of the data.

    Parameters
    ----------
    data : array, shape=([n_trials, ]n_channels, n_samples)
        Raw data.
    sfreq : float
        The sampling rate of the data.
    M : array, shape=(n_channels, n_channels)
        The Mixing matrix (as fitted with asr_calibrate).
    T : array, shape=(n_channels, n_channels)
        The Threshold matrix (as fitted with asr_calibrate).
    windowlen : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but short enough to allow for several 1000 windows to compute
        statistics over (default=0.5).
    lookahead:
        Amount of look-ahead that the algorithm should use. Since the
        processing is causal, the output signal will be delayed by this
        amount. This value is in seconds and should be between 0 (no
        lookahead) and WindowLength/2 (optimal lookahead). The recommended
        value is WindowLength/2. Default: 0.25
    stepsize:
        The steps in which the algorithm will be updated. The larger this is,
        the faster the algorithm will be. The value must not be larger than
        WindowLength * SamplingRate. The minimum value is 1 (update for every
        sample) while a good value would be sfreq//3. Note that an update
        is always performed also on the first and last sample of the data
        chunk. Default: 32
    max_dims : float, int
        Maximum dimensionality of artifacts to remove. This parameter
        denotes the maximum number of dimensions which can be removed from
        each segment. If larger than 1, `int(max_dims)` will denote the
        maximum number of dimensions removed from the data. If smaller than 1,
        `max_dims` describes a fraction of total dimensions. Defaults to 0.66.
    ab : 2-tuple | None
        Coefficients (A, B) of an IIR filter that is used to shape the
        spectrum of the signal when calculating artifact statistics. The
        output signal does not go through this filter. This is an optional way
        to tune the sensitivity of the algorithm to each frequency component
        of the signal. The default filter is less sensitive at alpha and beta
        frequencies and more sensitive at delta (blinks) and gamma (muscle)
        frequencies. Defaults to None.
    R : array, shape=(n_channels, n_channels)
        Previous reconstruction matrix. Defaults to None.
    Zi : array
        Previous filter conditions. Defaults to None.
    cov : array, shape=([n_trials, ]n_channels, n_channels) | None
        Covariance. If None (default), then it is computed from ``X_filt``.
        If a 3D array is provided, the average covariance is computed from
        all the elements in it. Defaults to None.
    carry :
        Initial portion of the data that will be added to the current data.
        If None, data will be interpolated. Defaults to None.
    return_states : bool
        If True, returns a dict including the updated states {"M":M, "T":T,
        "R":R, "Zi":Zi, "cov":cov, "carry":carry}. Defaults to False.
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average. For now, only
        euclidean ASR is supported.
    mem_splits : int
        Split the array in `mem_splits` segments to save memory.


    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Clean data.
    state : dict
        Output ASR parameters {"M":M, "T":T, "R":R, "Zi":Zi, "cov":cov,
        "carry":carry}.

    """
    if method == "riemann":
        warnings.warn("Riemannian ASR is not yet supported. Switching back to"
                      " Euclidean ASR.")
        method == "euclid"

    # calculate the the actual max dims based on the fraction parameter
    if maxdims < 1:
        maxdims = np.round(len(data) * maxdims)

    # set initial filter conditions of none was passed
    if Zi is None:
        _, Zi = yulewalk_filter(data, ab=ab, sfreq=sfreq,
                                zi=np.ones([len(data), 8]))

    # set the number of channels
    C, S = data.shape

    # set the number of windows
    N = np.round(windowlen * sfreq).astype(int)
    P = np.round(lookahead * sfreq).astype(int)

    # interpolate a portion of the data if no buffer was given
    if carry is None:
        carry = np.tile(2 * data[:, 0],
                        (P, 1)).T - data[:, np.mod(np.arange(P, 0, -1), S)]
    data = np.concatenate([carry, data], axis=-1)

    # splits = np.ceil(C*C*S*8*8 + C*C*8*s/stepsize + C*S*8*2 + S*8*5)...
    splits = mem_splits  # TODO: use this for parallelization MAKE IT A PARAM FIRST

    # loop over smaller segments of the data (for memory purposes)
    last_trivial = False
    last_R = None
    for i in range(splits):

        # set the current range
        i_range = np.arange(i * S // splits,
                            np.min([(i + 1) * S // splits, S]),
                            dtype=int)

        # filter the current window with yule-walker
        X, Zi = yulewalk_filter(data[:, i_range + P], sfreq=sfreq,
                                zi=Zi, ab=ab, axis=-1)

        # compute a moving average covariance
        Xcov, cov = \
            ma_filter(N,
                      np.reshape(np.multiply(np.reshape(X, (1, C, -1)),
                                             np.reshape(X, (C, 1, -1))),
                                 (C * C, -1)), cov)

        # set indices at which we update the signal
        update_at = np.arange(stepsize,
                              Xcov.shape[-1] + stepsize - 2,
                              stepsize)
        update_at = np.minimum(update_at, Xcov.shape[-1]) - 1

        # set the previous reconstruction matrix if none was assigned
        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = np.eye(C)

        Xcov = np.reshape(Xcov[:, update_at], (C, C, -1))

        # loop through the updating intervals
        last_n = 0
        for j in range(len(update_at) - 1):

            # get the eigenvectors/values.For method 'riemann', this should
            # be replaced with PGA/ nonlinear eigenvalues
            D, V = np.linalg.eigh(Xcov[:, :, j])

            # determine which components to keep
            keep = np.logical_or(D < np.sum((T @ V)**2, axis=0),
                                 np.arange(C) + 1 < (C - maxdims))
            trivial = np.all(keep)

            # set the reconstruction matrix (ie. reconstructing artifact
            # components using the mixing matrix)
            if not trivial:
                inv = pinv(np.multiply(keep[:, np.newaxis], V.T @ M))
                R = np.real(M @ inv @ V.T)
            else:
                R = np.eye(C)

            # apply the reconstruction
            n = update_at[j] + 1
            if (not trivial) or (not last_trivial):

                subrange = i_range[np.arange(last_n, n)]

                # generate a cosine signal
                blend_x = np.pi * np.arange(1, n - last_n + 1) / (n - last_n)
                blend = (1 - np.cos(blend_x)) / 2

                # use cosine blending to replace data with reconstructed data
                tmp_data = data[:, subrange]
                data[:, subrange] = np.multiply(blend, R @ tmp_data) + \
                                    np.multiply(1 - blend, last_R @ tmp_data) # noqa

            # set the parameters for the next iteration
            last_n, last_R, last_trivial = n, R, trivial

    # assign a new lookahead portion
    carry = np.concatenate([carry, data[:, -P:]])
    carry = carry[:, -P:]

    if return_states:
        return data[:, :-P], {"M": M, "T": T, "R": R, "Zi": Zi,
                              "cov": cov, "carry": carry}
    else:
        return data[:, :-P]


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1):
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
        The minimum and maximum standard deviations within which the power of
        a channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).

    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.

    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but not shorter than half a cycle of the high-pass filter that was
        used. Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).

    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).

    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.int_(np.round(np.arange(0, ns - N, (N * (1 - win_overlap)))))
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):

        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(np.int(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + np.int(max_bad_chans - 1), :] < np.min(zthresholds))

    # combine the two thresholds
    remove_mask = np.logical_or.reduce((mask1, mask2))
    removed_wins = np.where(remove_mask)

    # reconstruct the samples to remove
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

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, 1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        sample_mask = np.ones((1, ns), dtype=bool)

    return clean, sample_mask
