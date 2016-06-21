import numpy as np
from sklearn.base import BaseEstimator
from scipy import linalg
from .feature import EventsBinarizer, DataDelayer
from ..externals.six import string_types
from ..utils import warn
from sklearn.base import is_regressor


class EventRelatedRegressor(object):
    def __init__(self, raw, events, est=None, event_id=None, tmin=-.5, tmax=.5,
                 preproc_x=None, preproc_y=None):
        if events.shape[-1] != 3:
            raise ValueError('Events must be shape (n_events, 3)')
        if raw.preloaded is False:
            raise ValueError('Data must be preloaded')

        # Create events representation
        self.ev_ixs = events[:, 0]
        self.ev_types = events[:, 2]
        self.event_id = event_id
        binarizer = EventsBinarizer(raw.n_times, sfreq=raw.info['sfreq'])
        self.X = binarizer.fit_transform(self.ev_ixs, self.ev_types,
                                         self.event_id)

        # Prep output data
        self.raw = raw

        # Prepare data preprocessors + design matrix
        delayer = DataDelayer(time_window=[tmin, tmax],
                              sfreq=sig_cont.info['sfreq'])
        _check_preproc(preproc_x)
        _check_preproc(preproc_y)
        if preproc_x is not None:
            # Add the delays to the end of preproc_x
            preproc_x = Pipeline(preproc_x.steps + [('delayer', DataDelayer)])

        # Create model and attributes
        self.est = EncodingModel(est, preproc_x=preproc_x, preproc_y=preproc_y)
        self.tmin = tmin
        self.tmax = tmax

    def fit(self, picks=None):
        if picks is None:
            picks = np.arange(len(self.raw.ch_names))
        Y = self.raw._data[picks]
        self.est.fit(self.X, Y)
        self.coef_ = self.est.coef_


class EncodingModel(object):
    def __init__(self, est=None, preproc_x=None, preproc_y=None):
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

        References
        ----------
        .. [1] Smith, N. J., & Kutas, M. (2015). Regression-based estimation of
               ERP waveforms: II. Non-linear effects, overlap correction, and
               practical considerations. Psychophysiology, 52(2), 169-189.
        .. [2] Theunissen, F. E. et al. Estimating spatio-temporal receptive
               fields of auditory and visual neurons from their responses to
               natural stimuli. Network 12, 289-316 (2001).
        .. [3] Willmore, B. & Smyth, D. Methods for first-order kernel
               estimation: simple-cell receptive fields from responses to
               natural scenes. Network 14, 553-77 (2003).
        """
        self.est = _check_estimator(est)
        self.preproc_y = _check_preproc(preproc_y)
        self.preproc_x = _check_preproc(preproc_x)

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
        if X.shape[-1] != y.shape[-1]:
            raise ValueError('X and y must have last dimension same length.')
        n_chs, n_times = y.shape

        # Prepare the input feature matrix
        if self.preproc_x is not None:
            X = self.preproc_x.fit_transform(X)
        if self.preproc_y is not None:
            y = self.preproc_y.fit_transform(y)
        n_features = X.shape[0]

        # Fit the model and assign coefficients
        self.est.fit(X.T, y.T)
        self.coef_ = self.est._final_estimator.coef_
        self.X = X
        self.y = y

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
        return self.est.predict(X.T)


def _check_estimator(est):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    if est is None:
        est = Ridge()
    if not isinstance(est, str) and not is_regressor(est):
        warn("Custom estimators should have a `fit` and `predict` method,"
             " and should produce continuous output")

    elif isinstance(est, string_types):
        if est not in estimator_dict.keys():
            raise ValueError("No such solver: {0}".format(est))
        est = estimator_dict[est]

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


# Custom string-supported estimators
class CholeskySolver(BaseEstimator):
    def fit(self, X, y):
        a = (X.T * X).toarray()  # dot product of sparse matrices
        self.coef_ = linalg.solve(a, X.T * y.T, sym_pos=True,
                                  overwrite_a=True, overwrite_b=True).T

    def predict(self, X):
        return np.dot(self.coef_, X)

estimator_dict = dict(cholesky=CholeskySolver)
