import numpy as np
from sklearn.base import BaseEstimator
from scipy import linalg
from .feature import EventsBinarizer, DataDelayer
from ..io.pick import pick_types, pick_info
from ..externals.six import string_types
from ..utils import warn
from ..evoked import EvokedArray
from sklearn.base import is_regressor
from sklearn.pipeline import Pipeline


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
    est : None | instance of sklearn estimator
        A sklearn-style estimator. The input matrix will be a binary matrix
        of events, while the output will be the neural signal of each channel.
    event_id : dictionary
        A dictionary of (event_id: event_num) pairs
    tmin : float
        The minimum time for the rER[P/F]
    tmax : float
        The maximum time for the rER[P/F]
    preprox_x : None | instance of sklearn pipeline
        A pipeline that will be called on the input (X) variable.
        Must have a `fit_transform` method.
    preproc_y : None | instance of sklearn pipeline
        A pipeline that will be called on the output (y) variable.
        Must have a `fit_transform` method.
    preproc_func_xy : None | callable
        A function that will be called with *both* X and y as inputs. This is
        ideal for situations where both X and y should be modified in the same
        function. It will be called after the two preprocessing pipelines.

    References
    ----------
    .. [1] Smith, N. J., & Kutas, M. (2015). Regression-based estimation of
       ERP waveforms: II. Non-linear effects, overlap correction, and
       practical considerations. Psychophysiology, 52(2), 169-189.
    """
    def __init__(self, raw, events, est=None, event_id=None, tmin=-.1, tmax=.5,
                 preproc_x=None, preproc_y=None, preproc_func_xy=None,
                 picks=None):
        if events.shape[-1] != 3:
            raise ValueError('Events must be shape (n_events, 3)')
        if raw.preload is False:
            raise ValueError('Data must be preloaded')

        tmin_flip = -1 * tmax
        tmax_flip = -1 * tmin

        # Create events representation, pulling only events given in event_dict
        if event_id is None:
            event_id = dict(('%s' % i, i)
                            for i in range(len(np.unique(events[:, 2]))))
        events[:, 0] = events[:, 0] - raw.first_samp
        msk_keep = np.array([ii in event_id.values() for ii in events[:, 2]])
        events = events[msk_keep]

        self.ev_ixs = events[:, 0]
        self.ev_types = events[:, 2]
        self.unique_ev_types = np.unique(self.ev_types)
        self.event_id = event_id
        binarizer = EventsBinarizer(raw.n_times, sfreq=1., sparse=True)
        self.ev_binary = binarizer.fit_transform(self.ev_ixs, self.ev_types,
                                                 self.event_id)
        self.ev_names = binarizer.names_

        # Prep output data
        self.raw = raw
        if picks is None:
            picks = pick_types(raw.info, meg=True, eeg=True, ref_meg=True)
        self.picks = picks

        # Prepare data preprocessors + design matrix
        delayer = DataDelayer(time_window=[tmin_flip, tmax_flip],
                              sfreq=self.raw.info['sfreq'])
        self.delayer = delayer

        # Prep preprocessing chains
        _check_preproc(preproc_x)
        _check_preproc(preproc_y)
        if preproc_x is not None:
            # Add the delays to the end of preproc_x
            preproc_x = Pipeline(preproc_x.steps + [('delayer', delayer)])
        else:
            preproc_x = Pipeline([('delayer', delayer)])

        # Create model and attributes
        self.enc = EncodingModel(est, preproc_x=preproc_x, preproc_y=preproc_y,
                                 preproc_func_xy=preproc_func_xy)
        self.tmin = tmin
        self.tmax = tmax

    def fit(self, X=None, y=None):
        """Fit the encoding model."""
        self.raw.info = pick_info(self.raw.info, self.picks)
        Y = self.raw._data[self.picks]
        self.enc.fit(self.ev_binary, Y.T)
        return self

    def to_evoked(self, coef_name='coef_'):
        """Return model coefficients as Evoked objects.

        Parameters
        ----------
        coef_name : string
            The name of the fitted coefficients in the final step of the
            model's estimator attribute. For example, if the final step is
            an instance of `Ridge`, then this should be `coef_`. It will be
            used to retrieve the coefficients.

        Returns
        -------
        evokeds : dictionary of Evoked instances
            A dictionary of (event_id: evoked) pairs. The evoked obects are
            the rER[P/F] for each channel for that event.
        """
        if not hasattr(self.enc.est._final_estimator, coef_name):
            raise ValueError('Estimator either is not fit or does not use'
                             ' coefficient name: %s' % coef_name)
        # Unstack coefficients so they're shape (n_chans, n_feats, n_lags)
        n_delays = len(self.delayer.delays)
        coefs = getattr(self.enc.est._final_estimator, coef_name)
        coefs = np.stack([icoef.reshape([-1, n_delays]) for icoef in coefs])
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


class EncodingModel(object):
    def __init__(self, est=None, preproc_x=None, preproc_y=None,
                 preproc_func_xy=None):
        """Base structure for encoding models of neural signals.

        Fit an encoding model using arbitrary input transformations and a
        custom estimator.

        Parameters
        ----------
        est : instance of sklearn-style estimator
            The estimator to use for fitting. This is any object that contains
            a `fit` and `predict` method, which takes inputs of the form
            (X, y), and which creates a `.coef_` attribute upon fitting.
        preproc_x : instance of sklearn-style pipeline | None
            An object for preprocessing / transforming input data before the
            call to `est`. If None, no preprocessing will occur.
        preproc_y : instance of sklearn-style pipeline | None
            An object for preprocessing / transforming the output data before
            the call to `est`. If None, no preprocessing will occur.
        preproc_func_xy : callable | None
            A function to call after preprocessing pipelines have been run on X
            and y, but *before* the call to `fit` for the encoder. Should take
            two parameters: X and y, and return two parameters corresponding to
            X and y after the function has been applied.

        References
        ----------
        .. [1] Theunissen, F. E. et al. Estimating spatio-temporal receptive
               fields of auditory and visual neurons from their responses to
               natural stimuli. Network 12, 289-316 (2001).
        .. [2] Willmore, B. & Smyth, D. Methods for first-order kernel
               estimation: simple-cell receptive fields from responses to
               natural scenes. Network 14, 553-77 (2003).
        """
        self.est = _check_estimator(est)
        self.preproc_y = _check_preproc(preproc_y)
        self.preproc_x = _check_preproc(preproc_x)
        self.preproc_func_xy = preproc_func_xy

    def fit(self, X, y=None):
        """Fit the model.

        Fits a receptive field model. Model results are stored as attributes.

        Parameters
        ----------
        X : array, shape (n_times, n_features)
            The data on which we want to fit a regression model.
        y : array, shape (n_times, n_channels)

        Attributes
        ----------
        coefs_ : array, shape (n_channels, n_features)
            The coefficients fit on each row of X
        coef_names : array, shape (n_features,)
            A list of coefficient names, useful for keeping track of time lags
        """
        if y is None:
            raise ValueError('Must supply an output variable `y`')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have last dimension same length.')
        n_times, n_chs = y.shape

        # Prepare the input feature matrix
        if self.preproc_x is not None:
            X = self.preproc_x.fit_transform(X)
        if self.preproc_y is not None:
            y = self.preproc_y.fit_transform(y)

        if self.preproc_func_xy is not None:
            X, y = self.preproc_func_xy(X, y)

        # Fit the model and assign coefficients
        self.est.fit(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, preproc_x=None):
        """Generate predictions using fit coefficients.

        This uses the predict method of the final estimator in the
        `est` attribute.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input feature array.
        """
        # Preprocess X with the pipeline
        if preproc_x is not None:
            X = preproc_x.fit_transform(X)
        return self.est.predict(X)


def _check_estimator(est):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    if est is None:
        est = Ridge()
    elif isinstance(est, string_types):
        if est not in estimator_dict.keys():
            raise ValueError("No such solver: {0}".format(est))
        est = estimator_dict[est]()

    if not is_regressor(est):
        warn("Custom estimators should have a `fit` and `predict` method,"
             " and should produce continuous output")

    # Make sure we have a pipeline
    if not isinstance(est, Pipeline):
        est = Pipeline([('est', est)])
    return est


def _check_preproc(est):
    from sklearn.pipeline import Pipeline
    if est is None:
        return est
    if not isinstance(est, Pipeline):
        raise ValueError('preproc must be a sklearn Pipeline or None')
    if not hasattr(est, 'transform'):
        raise ValueError('preproc must have a transform method')
    return est


def pull_feature_names(est, name_attr='names_', base_names=None):
    """Iterate through steps of a Pipeline and extract feature names.

    Incomplete, but something like this could be used to keep
    track of feature expansions that took place in a Pipeline (e.g., lags)
    """
    if not isinstance(est, Pipeline):
        raise ValueError('est must be an instance of Pipeline')
    all_names = []
    for nm, step in est.steps:
        if hasattr(step, name_attr):
            all_names.append(getattr(step, name_attr))
    return all_names


# Custom string-supported estimators
class CholeskySolver(BaseEstimator):
    """Solves a linear system with Cholesky decomposition.

    This will work with sparse inputs to save memory.
    """
    def __init__(self):
        self._estimator_type = 'regressor'  # For sklearn compatibility

    def fit(self, X, y=None):
        """Fit the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input array to fit.
        y : array, shape (n_samples, n_outputs)
            The output array to fit.
        """
        a = (X.T * X).toarray()  # dot product of sparse matrices

        self.coef_ = linalg.solve(a, X.T * y, sym_pos=True,
                                  overwrite_a=True, overwrite_b=True).T
        return self

    def predict(self, X):
        return np.dot(self.coef_, X)

estimator_dict = dict(cholesky=CholeskySolver)
