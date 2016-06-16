import numpy as np
from sklearn.base import BaseEstimator
from scipy import linalg
from ..externals.six import string_types


class EncodingModel(object):
    def __init__(self, est=None, delays=[0.], preproc_x=None, preproc_y=None):
        """Fit a STRF model.

        Fit an encoding model using time lags and a custom estimator.
        Compatible either with continuous or events-based input
        stimuli. It creates time lags for the input matrix, performs
        preprocessing according to any transformers in the pipeline in `est`,
        and then fits a model with the final estimator in `est`.

        Parameters
        ----------
        est : instance of sklearn-style estimator
            The estimator to use for fitting. This is any object that contains
            a `fit` and `predict` method, which takes inputs of the form
            (X, y), and which creates a `.coef_` attribute upon fitting.
        delays : array, shape (n_delays,)
            The delays to include when creating time lags. The input array X
            will end up having shape (n_feats * n_delays, n_times)
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
        self.delays = np.asarray(delays)
        self.n_delays = len(self.delays)
        self.est = _check_estimator(est)
        self.preproc_y = _check_preproc(preproc_y)
        self.preproc_x = _check_preproc(preproc_x)

    def fit(self, X, y=None):
        """Fit the model.

        Fits a receptive field model. Model results are stored as attributes.

        Parameters
        ----------
        raw : instance of MNE Raw
            The data on which we want to fit a regression model.
        events : array, shape (n_events, 3)
            An MNE events array specifying indices for event onsets
        event_id : dictionary | None
            A dictionary of (event_name: event_int) pairs, corresponding to the
            mapping from event name strings to the 3rd column of events.
        continuous : array, shape (n_feats, n_times)
            Continuous input features in the regression.
        continuous_names : array, shape (n_feats,) | None
            Names for the input continuous variables.
        picks : array, shape (n_picks,) | None
            Indices for channels to use in model fitting. If None, all channels
            will be fit.

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
        self.coef_ = np.zeros([n_chs, n_features])
        for ii in range(n_chs):
            i_y = y[ii][:, np.newaxis]
            self.est.fit(X.T, i_y)
            self.coef_[ii, :] = self.est.steps[-1][-1].coef_
        self.X = X
        self.y = y

    def predict(self, X, preproc_x=None):
        """Generate predictions using fit coefficients.

        This uses the `coef_` attribute for predictions.
        """
        if preproc_x is not None:
            X = preproc_x.fit_transform(X)
        X = self.est._pre_transform(X.T)[0].T
        preds = np.dot(self.coef_, X)
        return preds


def _check_estimator(est):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    if est is None:
        est = Ridge()
    elif isinstance(est, string_types):
        if est not in estimator_dict.keys():
            raise ValueError("No such solver: {0}".format(est))
        est = estimator_dict[est]

    # Make sure we have a pipeline
    if not isinstance(est, Pipeline):
        est = Pipeline([('est', est)])
    for imethod in ['fit', 'predict']:
        if not hasattr(est.steps[-1][-1], imethod):
            raise ValueError('estimator must have a %s method' % imethod)
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
