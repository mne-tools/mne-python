"""Base and high-level classes for fitting encoding models."""
# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.de>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import sparse
from ..io.pick import pick_types, pick_info
from ..externals.six import string_types
from ..evoked import EvokedArray


class EventRelatedRegressor(object):
    """Fit an rER[R/F] model to Raw objects.

    This uses event onsets to create a continuous, binary representation of
    event times. It creates a time-lagged version of these events, in order to
    fit a set coefficients per event type that represent the effect of that
    event on the signal of interest.

    Parameters
    ----------
    raw : instance of Raw
        The data to be modeled with event onsets. Data must be preloaded.
    events : array, shape (n_events, 3)
        An MNE events array specifying event onset indices (first column),
        and event types (last column)
    event_id : dictionary | None
        An optional dictionary of (event_id: event_num) pairs. Used for
        labeling event types. Defaults to None.
    covariates : array, shape (n_events, n_covariates)
        An optional array of covariate (e.g., continuous) values for each
        event. Defaults to None.
    covariate_names : array, shape (n_covariates,)
        An optional list of names for each covariate. If None, they will be
        generated automatically. Defaults to None.
    est : None | instance of sklearn estimator | string
        A sklearn-style estimator. The input matrix will be a binary matrix
        of events, while the output will be the neural signal of each channel.
        If a string, then an instance of a sklearn `Ridge` model will be
        created with alpha == 0, and this parameter passed to the `solver`
        parameter.
    tmin : float
        The minimum time for the rER[P/F]
    tmax : float
        The maximum time for the rER[P/F]
    coef_name : string
        The name of the fitted coefficients in the final step of the
        model's estimator attribute. For example, if the final step is
        an instance of `Ridge`, then this should be `coef_`. It will be
        used to retrieve the coefficients.
    remove_outliers : bool
        Whether to clean data samples based on outliers. Defaults to True.
    remove_empty : bool
        Whether to remove samples where there is no event in X.

    References
    ----------
    .. [1] Smith, N. J., & Kutas, M. (2015). Regression-based estimation of
       ERP waveforms: II. Non-linear effects, overlap correction, and
       practical considerations. Psychophysiology, 52(2), 169-189.
    """
    def __init__(self, raw, events, event_id=None, covariates=None,
                 covariate_names=None, est=None, tmin=-.1, tmax=.5, picks=None,
                 coef_name='coef_', remove_outliers=True, remove_empty=True):
        from .feature import EventsBinarizer, FeatureDelayer

        if events.shape[-1] != 3:
            raise ValueError('Events must be shape (n_events, 3)')
        if raw.preload is False:
            raise ValueError('Data must be preloaded')

        # Pull event information
        if event_id is None:
            event_id = dict(('%s' % i, i)
                            for i in range(len(np.unique(events[:, 2]))))
        events = events.copy()
        events[:, 0] = events[:, 0] - raw.first_samp
        msk_keep = np.array([ii in event_id.values() for ii in events[:, 2]])
        events = events[msk_keep]
        self.ev_ixs, _, self.ev_types = events.T
        self.unique_ev_types = np.unique(self.ev_types)
        self.event_id = event_id

        # Create events representation, pulling only events given in event_dict
        binarizer = EventsBinarizer(raw.n_times, sfreq=1., sparse=True)
        self.ev_binary = binarizer.fit_transform(self.ev_ixs, self.ev_types,
                                                 self.event_id, )
        self.ev_names = binarizer.names_

        # Prepare output data
        self.raw = raw
        if picks is None:
            picks = pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        self.picks = picks

        # Prepare data preprocessors + design matrix
        # Rev + mult by -1 to make it in "times" not "lags"
        time_window = -1 * np.array([tmax, tmin])
        delayer = FeatureDelayer(time_window=time_window,
                                 sfreq=self.raw.info['sfreq'])
        self.delayer = delayer
        self.tmin = tmin
        self.tmax = tmax

        # Create model, and cause the estimator to mask before fit
        self.est = _check_estimator(est)
        self.coef_name = coef_name
        self.est.steps = [('delayer', delayer)] + self.est.steps
        self.remove_empty = remove_empty
        self.remove_outliers = remove_outliers
        if remove_empty is True:
            # Replace the final estimator w/ a masker version of this
            est_tuple = self.est.steps[-1]
            self.est.steps[-1] = (est_tuple[0], SampleMasker(
                est_tuple[1], mask_val=0))

    def fit(self, X=None, y=None):
        """Fit the encoding model."""
        self.raw.info = pick_info(self.raw.info, self.picks)
        X = self.ev_binary
        Y = self.raw._data[self.picks].T
        if self.remove_outliers is True:
            X, Y = remove_outliers(X, Y)
        self.est.fit(X, Y)
        return self

    def to_evoked(self):
        """Return model coefficients as Evoked objects.

        Returns
        -------
        evokeds : dictionary of `Evoked` instances
            A dictionary of (event_id: `Evoked`) pairs. The `Evoked` objects
            are the rER[P/F] for each channel for that event.
        """
        # Unstack coefficients so they're shape (n_chans, n_feats, n_lags)
        n_delays = len(self.delayer.delays)

        est = get_final_est(self.est)
        coefs = get_coefs(est, self.coef_name)
        coefs = np.array([icoef.reshape(-1, n_delays) for icoef in coefs])
        # Reverse last dimension so that it is in time, not lags
        coefs = coefs[..., ::-1]
        # Iterate through event types and create the evoked objects
        evokeds = dict()
        for ev_name, ev_num in self.event_id.items():
            n_ev = (self.ev_types == ev_num).sum()
            coefs_ix = np.where(np.asarray(self.ev_names) == ev_name)[0][0]

            # nave and kind are technically incorrect
            evokeds[ev_name] = EvokedArray(
                coefs[:, coefs_ix, :], info=self.raw.info, comment=ev_name,
                tmin=self.tmin, nave=n_ev, kind='average')
        return evokeds


class SampleMasker(object):
    """Remove samples with a particular value before fitting.

    Parameters
    ----------
    estimator : instance of sklearn-style estimator
        The estimator to be called after samples are removed.
    mask_val : float, int, nan | callable.
        The value that will be removed from the dataset. If a callable,
        it should return a boolean mask of length n_samples that takes
        X an array of shape (n_samples, n_features) as input.
    ixs : array, shape (n_samples,)
        An optional array of indices for manually specifying which samples
        to keep during training. Defaults to None.
    ixs_pred : array, shape (n_samples,) | None
        An optional array of indices for manually specifying which samples
        to keep during prediction. Defaults to None.
    mask_condition : 'all', 'any'
        Whether only 1 `mask_val` will trigger removal of a row, or if
        all values of the row must be `mask_val` for removal.
    """
    def __init__(self, estimator, mask_val=None, ixs=None, ixs_pred=None,
                 mask_condition='all'):
        if all(ii is not None for ii in [mask_val, ixs]):
            raise ValueError('Supply one of mask_val | ixs, or both')
        if ixs is not None:
            ixs = np.asarray(ixs).astype(int)
        self.ixs = ixs
        if ixs_pred is not None:
            ixs_pred = np.asarray(ixs_pred).astype(int)
        self.ixs_pred = ixs_pred

        self.mask_val = mask_val
        if mask_condition not in ['all', 'any']:
            raise ValueError('condition must be one of "all" or "any"')
        self.mask_condition = mask_condition
        self.est = _check_estimator(estimator)

    def fit(self, X, y):
        """Remove datapoints then fit the estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Rows will be removed.
        y : array, shape (n_samples, n_targets)
            The output array for the model. Rows will be removed.
        """
        X, y = self.mask_data(X, y, ixs=self.ixs)
        self.est.fit(X, y)
        return self

    def predict(self, X):
        """Call self.est.predict.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Rows will NOT be removed.
        """
        X = self.mask_data(X, ixs=self.ixs_pred)
        return self.est.predict(X)

    def mask_data(self, X, y=None, ixs=None):
        """Remove datapoints according to indices or masked values.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array for the model. Rows will be removed.
        y : array, shape (n_samples, n_targets)
            The output array for the model. Rows will be removed.
        ixs : array, shape (n_samples_to_keep,)
            An optional array of indices to keep before removing
            any samples based on their values.

        Returns
        -------
        X : array, shape (n_samples_after_removal, n_features)
            The input X array after removing rows based on ixs, then
            based on the values in X.
        y : array, shape (n_samples_after_removal, n_targets) (optional)
            The output y array after removing rows based on ixs, then
            based on the values in X. Only returned if `y` is passed.
        """

        # First remove indices if we've manually specified them
        if y is not None:
            if y.ndim == 1:
                y = y[:, np.newaxis]
        if ixs is not None:
            if X.shape[0] < ixs.max():
                raise ValueError('ixs exceed data shape')
            X = X[ixs]
            if y is not None:
                y = y[ixs]

        # Find data points with value we want to remove
        if self.mask_val is not None:
            if hasattr(self.mask_val, '__call__'):
                # If a callable, pass X as an input, assume bool output
                mask = self.mask_val(X)
                if mask.ndim > 1:
                    raise ValueError('Output mask must be shape (n_samples,)')
            else:
                if self.mask_val is np.nan:
                    mask = ~np.isnan(X)
                else:
                    mask = X != self.mask_val

                # Decide which rows to remove
                if self.mask_condition == 'all':
                    # Remove rows w/ NO good values
                    mask = mask.sum(axis=1) == 0
                else:
                    # Remove rows w/ AT LEAST 1 bad value
                    mask = mask.sum(axis=1) < mask.shape[-1]

            # Now change mask to rows that we wish to *keep*
            self.mask = ~mask
            mask_ixs = np.where(self.mask)[0]
            mask_ixs = np.asarray(mask_ixs).squeeze()  # In case its a matrix
            X = X[mask_ixs, :]
            if y is not None:
                y = y[mask_ixs, :]
        else:
            self.mask = np.ones(X.shape[0], dtype=bool)
        # Return masked values
        if y is not None:
            return X, y
        return X

    def transform(self, X, y=None):
        """Return a subset of data points.

        Returns
        -------
        data : array, shape (n_features, n_ixs)
            A subset of rows from the input array.
        """
        return self.mask_data(X, y)


def remove_outliers(X, y, reject=None, flat=None, info=None, tstep=None):
    """Remove data points based on peak to peak amplitude.

    Parameters
    ----------
    X : array, shape (n_times, n_features)
        The input array (usually stimuli or continuous events)
    y : array, shape (n_times, n_channels)
        The neural data. Will be used for detecting noisy datapoints
    reject : None | dict
        For cleaning raw data before the regression is performed: set up
        rejection parameters based on peak-to-peak amplitude in continuously
        selected subepochs. If None, no rejection is done.
        If dict, keys are types ('grad' | 'mag' | 'eeg' | 'eog' | 'ecg')
        and values are the maximal peak-to-peak values to select rejected
        epochs, e.g.::

            reject = dict(grad=4000e-12, # T / m (gradiometers)
                          mag=4e-11, # T (magnetometers)
                          eeg=40e-5, # V (EEG channels)
                          eog=250e-5 # V (EOG channels))

    flat : None | dict
        or cleaning raw data before the regression is performed: set up
        rejection parameters based on flatness of the signal. If None, no
        rejection is done. If a dict, keys are ('grad' | 'mag' |
        'eeg' | 'eog' | 'ecg') and values are minimal peak-to-peak values to
        select rejected epochs.
    info : None | instance of MNE Info object
        The Info object corresponding to the neural data `y`.
    tstep : float
        Length of windows for peak-to-peak detection for raw data cleaning.

    Returns
    -------
    X_cleaned : array, shape (n_clean_times, n_features)
        The input X array w/ non-active and noisy rows removed.
    y_cleaned : array, shape (n_clean_times, n_channels)
        The input y array w/ non-active and noisy rows removed.
    """
    from ..utils import _reject_data_segments

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y have different numbers of timepoints')
    is_sparse = True if isinstance(X, sparse.spmatrix) else False

    # reject positions based on extreme steps in the data
    keep_rows = np.arange(X.shape[0])
    if reject is not None:
        _, inds_bad = _reject_data_segments(y.T, reject, flat, decim=None,
                                            info=info, tstep=tstep)
        # Expand to include all bad indices, and remove from rows to keep
        inds_bad = np.hstack(range(t0, t1) for t0, t1 in inds_bad)
        keep_rows = np.setdiff1d(keep_rows, inds_bad)

    if is_sparse is True:
        X = X.tocsr()
    return X[keep_rows], y[keep_rows]


def get_final_est(est):
    """Return the final component of a sklearn estimator/pipeline.

    Parameters
    ----------
    est : pipeline, or sklearn / MNE estimator
        An estimator chain from which you wish to pull the final estimator.

    Returns
    -------
    est : the sklearn-style estimator at the end of the input chain.
    """
    # Define classes where we pull `est` manually
    import sklearn
    from distutils.version import LooseVersion
    if LooseVersion(sklearn.__version__) < '0.16':
        raise ValueError('Encoding models require sklearn version >= 0.16')

    iter_classes = (SampleMasker,)
    # Iterate in case we have meta estimators w/in meta estimators
    while hasattr(est, '_final_estimator') or isinstance(est, iter_classes):
        if isinstance(est, iter_classes):
            est = est.est
        else:
            est = est._final_estimator
    return est


def get_coefs(est, coef_name='coef_'):
    """Pull coefficients from an estimator object.

    Parameters
    ----------
    est : a sklearn estimator
        The estimator from which to pull the coefficients.
    coef_name : string
        The name of the attribute corresponding to the coefficients created
        after fitting. Defaults to 'coef_'

    Returns
    -------
    coefs : array, shape (n_targets, n_coefs)
        The output coefficients.
    """
    if not hasattr(est, coef_name):
        raise ValueError('Estimator either is not fit or does not use'
                         ' coefficient name: %s' % coef_name)
    coefs = getattr(est, coef_name)
    return coefs


def _check_estimator(est):
    """Ensure the estimator will work for regression data."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    # Define string-based solvers
    _ridge_solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']

    if est is None:
        est = Ridge(alpha=0)
    elif isinstance(est, string_types):
        if est not in _ridge_solvers:
            raise ValueError("No such solver: {0}\nAllowed solvers are:"
                             " {1}".format(est, _ridge_solvers))
        est = Ridge(alpha=0, solver=est, fit_intercept=False)

    reqd_attributes = ['fit', 'predict']
    for attr in reqd_attributes:
        if not hasattr(est, attr):
            raise ValueError('Estimator does not have a %s method' % attr)

    # Make sure we have a pipeline
    if not isinstance(est, Pipeline):
        est = Pipeline([('est', est)])
    return est


def _check_preproc(est):
    """Ensure that the estimator is a Pipeline w/ transforms."""
    from sklearn.pipeline import Pipeline
    if est is None:
        pass
    elif not isinstance(est, Pipeline):
        raise ValueError('preproc must be a sklearn Pipeline or None')
    elif not hasattr(est, 'transform'):
        raise ValueError('preproc must have a transform method')
    return est
