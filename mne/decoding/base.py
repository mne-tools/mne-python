"""Base class copy from sklearn.base."""
# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..fixes import BaseEstimator
from ..utils import check_version, deprecated


class LinearModel(BaseEstimator):
    """Compute and store patterns from linear models.

    The linear model coefficients (filters) are used to extract discriminant
    neural sources from the measured data. This class computes the
    corresponding patterns of these linear filters to make them more
    interpretable [1]_.

    Parameters
    ----------
    model : object | None
        A linear model from scikit-learn with a fit method
        that updates a ``coef_`` attribute.
        If None the model will be LogisticRegression.

    Attributes
    ----------
    ``filters_`` : ndarray
        If fit, the filters used to decompose the data.
    ``patterns_`` : ndarray
        If fit, the patterns used to restore M/EEG signals.

    Notes
    -----
    .. versionadded:: 0.10

    See Also
    --------
    CSP
    mne.preprocessing.ICA
    mne.preprocessing.Xdawn

    References
    ----------
    .. [1] Haufe, S., Meinecke, F., Gorgen, K., Dahne, S., Haynes, J.-D.,
           Blankertz, B., & Biebmann, F. (2014). On the interpretation of
           weight vectors of linear models in multivariate neuroimaging.
           NeuroImage, 87, 96-110.
    """

    def __init__(self, model=None):  # noqa: D102
        if model is None:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()

        self.model = model

    def fit(self, X, y):
        """Estimate the coefficients of the linear model.

        Save the coefficients in the attribute ``filters_`` and
        computes the attribute ``patterns_``.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples,)
            The target values.

        Returns
        -------
        self : instance of LinearModel
            Returns the modified instance.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('LinearModel only accepts 2-dimensional X, got '
                             '%s instead.' % (X.shape,))

        # fit the Model
        self.model.fit(X, y)

        # computes the patterns
        if (
            not hasattr(self.model, 'coef_') or  # missing attribute
            self.model.coef_.ndim > 2 or         # weird case
            (self.model.coef_.size not in
             self.model.coef_.shape)             # shape (n), (n, 1) or (1, n)
        ):

            raise ValueError('model needs a unidimensional coef_ attribute to '
                             'compute the patterns')
        self.filters_ = np.squeeze(self.model.coef_)
        self.patterns_ = np.cov(X.T).dot(self.filters_)

        return self

    def transform(self, X):
        """Transform the data using the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.
        """
        return self.model.transform(X)

    def fit_transform(self, X, y):
        """Fit the data and transform it using the linear model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The training input samples to estimate the linear coefficients.
        y : array, shape (n_samples,)
            The target values.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted targets.

        """
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """Compute predictions of y from X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data used to compute the predictions.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predictions.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """Score the linear model computed on the given test data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to transform.
        y : array, shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            Score of the linear model
        """
        return self.model.score(X, y)

    @deprecated('plot_filters is deprecated and will be removed in 0.15, '
                'use EvokedArray instead.')
    def plot_patterns(self, info, times=None, ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scale=None, scale_time=1e3, unit='a.u.',
                      res=64, size=1, cbar_fmt='%3.1f',
                      name_format='%01d ms', proj=False, show=True,
                      show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic patterns of the linear model.

        The patterns explain how the measured data was generated
        from the neural sources (a.k.a. the forward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used to fit the linear model.
            If not possible, consider using ``create_info``.
        times : float | array of floats | None.
            The time point(s) to plot. If None, the number of ``axes``
            determines the amount of time point(s). If ``axes`` is also None,
            10 topographies will be shown with a regular time spacing between
            the first and last time instant.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are
            collected in pairs and the RMS for each pair is plotted.
            If None, then first available channel type from order given
            above is used. Defaults to None.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to be
            specified for Neuromag data). If possible, the correct layout file
            is inferred from the data; if no appropriate layout file was found
            the layout is automatically generated from the sensor locations.
        vmin : float | callable
            The value specfying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specfying the upper bound of the color range.
            If None, the maximum absolute value is used. If vmin is None,
            but vmax is not, defaults to np.min(data).
            If callable, the output equals vmax(data).
        cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
            Colormap to use. If tuple, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging
            the colorbar with left and right mouse button. Left mouse button
            moves the scale up and down and right mouse button adjusts the
            range. Hitting space bar resets the range. Up and down arrows can
            be used to change the colormap. If None, 'Reds' is used for all
            positive data, otherwise defaults to 'RdBu_r'. If 'interactive',
            translates to (None, True). Defaults to 'RdBu_r'.

            .. warning::  Interactive mode works smoothly only for a small
                amount of topomaps.

        sensors : bool | str
            Add markers for sensor locations to the plot. Accepts matplotlib
            plot format string (e.g., 'r+' for red plusses). If True,
            a circle will be used (via .add_artist). Defaults to True.
        colorbar : bool
            Plot a colorbar.
        scale : dict | float | None
            Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
            for grad and 1e15 for mag.
        scale_time : float | None
            Scale the time labels. Defaults to 1e3.
        unit : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        name_format : str
            String format for topomap values. Defaults to "%03f ms"
        proj : bool | 'interactive'
            If true SSP projections are applied before display.
            If 'interactive', a check box for reversible selection
            of SSP projection vectors will be show.
        show : bool
            Show figure if True.
        show_names : bool | callable
            If True, show channel names on top of the map. If a callable is
            passed, channel names will be formatted using the callable; e.g.,
            to delete the prefix 'MEG ' from all channel names, pass the
            function lambda x: x.replace('MEG ', ''). If `mask` is not None,
            only significant sensors will be shown.
        title : str | None
            Title. If None (default), no title is displayed.
        mask : ndarray of bool, shape (n_channels, n_times) | None
            The channels to be marked as significant at a given time point.
            Indices set to `True` will be considered. Defaults to None.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                     linewidth=0, markersize=4)

        outlines : 'head' | 'skirt' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will
            be drawn. If 'skirt' the head scheme will be drawn, but sensors are
            allowed to be plotted outside of the head circle. If dict, each key
            refers to a tuple of x and y positions, the values in 'mask_pos'
            will serve as image mask, and the 'autoshrink' (bool) field will
            trigger automated shrinking of the positions due to points outside
            the outline. Alternatively, a matplotlib patch object can be passed
            for advanced masking options, either directly or as a function that
            returns patches (required for multi-axis plots). If None, nothing
            will be drawn. Defaults to 'head'.
        contours : int | False | None
            The number of contour lines to draw.
            If 0, no contours will be drawn.
        image_interp : str
            The image interpolation to be used.
            All matplotlib options are accepted.
        average : float | None
            The time window around a given time to be used for averaging
            (seconds). For example, 0.01 would translate into window that
            starts 5 ms before and ends 5 ms after a given time point.
            Defaults to None, which means no averaging.
        head_pos : dict | None
            If None (default), the sensors are positioned such that they span
            the head circle. If dict, can have entries 'center' (tuple) and
            'scale' (tuple) for what the center and scale of the head
            should be relative to the electrode locations.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        from .. import EvokedArray

        if times is None:
            tmin = 0
            times = 'auto'
        else:
            tmin = times[0]

        # create an evoked
        patterns = EvokedArray(self.patterns_.reshape(info['nchan'], -1),
                               info, tmin=tmin)
        # the call plot_topomap
        return patterns.plot_topomap(times=times, ch_type=ch_type,
                                     layout=layout, vmin=vmin, vmax=vmax,
                                     cmap=cmap, colorbar=colorbar, res=res,
                                     cbar_fmt=cbar_fmt, sensors=sensors,
                                     scale=scale, scale_time=scale_time,
                                     time_format=name_format, size=size,
                                     show_names=show_names, unit=unit,
                                     mask_params=mask_params,
                                     mask=mask, outlines=outlines,
                                     contours=contours, title=title,
                                     image_interp=image_interp, show=show,
                                     head_pos=head_pos, average=average)

    @deprecated('plot_filters is deprecated and will be removed in 0.15, '
                'use EvokedArray instead.')
    def plot_filters(self, info, times=None, ch_type=None, layout=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scale=None, scale_time=1e3, unit='a.u.',
                     res=64, size=1, cbar_fmt='%3.1f',
                     name_format='%01d ms', proj=False, show=True,
                     show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic filters of the linear model.

        The filters are used to extract discriminant neural sources from
        the measured data (a.k.a. the backward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used to fit the linear model.
            If not possible, consider using ``create_info``.
        times : float | array of floats | None.
            The time point(s) to plot. If None, the number of ``axes``
            determines the amount of time point(s). If ``axes`` is also None,
            10 topographies will be shown with a regular time spacing between
            the first and last time instant.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are
            collected in pairs and the RMS for each pair is plotted.
            If None, then first available channel type from order given
            above is used. Defaults to None.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to be
            specified for Neuromag data). If possible, the correct layout file
            is inferred from the data; if no appropriate layout file was found
            the layout is automatically generated from the sensor locations.
        vmin : float | callable
            The value specfying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specfying the upper bound of the color range.
            If None, the maximum absolute value is used. If vmin is None,
            but vmax is not, defaults to np.min(data).
            If callable, the output equals vmax(data).
        cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
            Colormap to use. If tuple, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging
            the colorbar with left and right mouse button. Left mouse button
            moves the scale up and down and right mouse button adjusts the
            range. Hitting space bar resets the range. Up and down arrows can
            be used to change the colormap. If None, 'Reds' is used for all
            positive data, otherwise defaults to 'RdBu_r'. If 'interactive',
            translates to (None, True). Defaults to 'RdBu_r'.

            .. warning::  Interactive mode works smoothly only for a small
                amount of topomaps.

        sensors : bool | str
            Add markers for sensor locations to the plot. Accepts matplotlib
            plot format string (e.g., 'r+' for red plusses). If True,
            a circle will be used (via .add_artist). Defaults to True.
        colorbar : bool
            Plot a colorbar.
        scale : dict | float | None
            Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
            for grad and 1e15 for mag.
        scale_time : float | None
            Scale the time labels. Defaults to 1e3.
        unit : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        name_format : str
            String format for topomap values. Defaults to "%03f ms"
        proj : bool | 'interactive'
            If true SSP projections are applied before display.
            If 'interactive', a check box for reversible selection
            of SSP projection vectors will be show.
        show : bool
            Show figure if True.
        show_names : bool | callable
            If True, show channel names on top of the map. If a callable is
            passed, channel names will be formatted using the callable; e.g.,
            to delete the prefix 'MEG ' from all channel names, pass the
            function lambda x: x.replace('MEG ', ''). If `mask` is not None,
            only significant sensors will be shown.
        title : str | None
            Title. If None (default), no title is displayed.
        mask : ndarray of bool, shape (n_channels, n_times) | None
            The channels to be marked as significant at a given time point.
            Indices set to `True` will be considered. Defaults to None.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                     linewidth=0, markersize=4)

        outlines : 'head' | 'skirt' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will
            be drawn. If 'skirt' the head scheme will be drawn, but sensors are
            allowed to be plotted outside of the head circle. If dict, each key
            refers to a tuple of x and y positions, the values in 'mask_pos'
            will serve as image mask, and the 'autoshrink' (bool) field will
            trigger automated shrinking of the positions due to points outside
            the outline. Alternatively, a matplotlib patch object can be passed
            for advanced masking options, either directly or as a function that
            returns patches (required for multi-axis plots). If None, nothing
            will be drawn. Defaults to 'head'.
        contours : int | False | None
            The number of contour lines to draw.
            If 0, no contours will be drawn.
        image_interp : str
            The image interpolation to be used.
            All matplotlib options are accepted.
        average : float | None
            The time window around a given time to be used for averaging
            (seconds). For example, 0.01 would translate into window that
            starts 5 ms before and ends 5 ms after a given time point.
            Defaults to None, which means no averaging.
        head_pos : dict | None
            If None (default), the sensors are positioned such that they span
            the head circle. If dict, can have entries 'center' (tuple) and
            'scale' (tuple) for what the center and scale of the head
            should be relative to the electrode locations.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        from .. import EvokedArray

        if times is None:
            tmin = 0
            times = 'auto'
        else:
            tmin = times[0]

        # create an evoked
        filters = EvokedArray(self.filters_.T.reshape(info['nchan'], -1),
                              info, tmin=tmin)
        # the call plot_topomap
        return filters.plot_topomap(times=times, ch_type=ch_type,
                                    layout=layout, vmin=vmin, vmax=vmax,
                                    cmap=cmap, colorbar=colorbar, res=res,
                                    cbar_fmt=cbar_fmt, sensors=sensors,
                                    scale=scale, scale_time=scale_time,
                                    time_format=name_format, size=size,
                                    show_names=show_names, unit=unit,
                                    mask_params=mask_params,
                                    mask=mask, outlines=outlines,
                                    contours=contours, title=title,
                                    image_interp=image_interp, show=show,
                                    head_pos=head_pos, average=average)


def _set_cv(cv, estimator=None, X=None, y=None):
    """Set the default CV depending on whether clf is classifier/regressor."""
    from sklearn.base import is_classifier

    # Detect whether classification or regression
    if estimator in ['classifier', 'regressor']:
        est_is_classifier = estimator == 'classifier'
    else:
        est_is_classifier = is_classifier(estimator)
    # Setup CV
    if check_version('sklearn', '0.18'):
        from sklearn import model_selection as models
        from sklearn.model_selection import (check_cv, StratifiedKFold, KFold)
        if isinstance(cv, (int, np.int)):
            XFold = StratifiedKFold if est_is_classifier else KFold
            cv = XFold(n_splits=cv)
        elif isinstance(cv, str):
            if not hasattr(models, cv):
                raise ValueError('Unknown cross-validation')
            cv = getattr(models, cv)
            cv = cv()
        cv = check_cv(cv=cv, y=y, classifier=est_is_classifier)
    else:
        from sklearn import cross_validation as models
        from sklearn.cross_validation import (check_cv, StratifiedKFold, KFold)
        if isinstance(cv, (int, np.int)):
            if est_is_classifier:
                cv = StratifiedKFold(y=y, n_folds=cv)
            else:
                cv = KFold(n=len(y), n_folds=cv)
        elif isinstance(cv, str):
            if not hasattr(models, cv):
                raise ValueError('Unknown cross-validation')
            cv = getattr(models, cv)
            if cv.__name__ not in ['KFold', 'LeaveOneOut']:
                raise NotImplementedError('CV cannot be defined with str for'
                                          ' sklearn < .017.')
            cv = cv(len(y))
        cv = check_cv(cv=cv, X=X, y=y, classifier=est_is_classifier)

    # Extract train and test set to retrieve them at predict time
    if hasattr(cv, 'split'):
        cv_splits = [(train, test) for train, test in
                     cv.split(X=np.zeros_like(y), y=y)]
    else:
        # XXX support sklearn.cross_validation cv
        cv_splits = [(train, test) for train, test in cv]

    if not np.all([len(train) for train, _ in cv_splits]):
        raise ValueError('Some folds do not have any train epochs.')

    return cv, cv_splits


def _check_estimator(estimator, get_params=True):
    """Check whether an object has the methods required by sklearn."""
    valid_methods = ('predict', 'transform', 'predict_proba',
                     'decision_function')
    if (
        (not hasattr(estimator, 'fit')) or
        (not any(hasattr(estimator, method) for method in valid_methods))
    ):
        raise ValueError('estimator must be a scikit-learn transformer or '
                         'an estimator with the fit and a predict-like (e.g. '
                         'predict_proba) or a transform method.')

    if get_params and not hasattr(estimator, 'get_params'):
        raise ValueError('estimator must be a scikit-learn transformer or an '
                         'estimator with the get_params method that allows '
                         'cloning.')


def _make_scorer(scoring):
    """Make scorer.

    Parameters
    ----------
    scoring : str | callable
        If str, must be compatible with sklearn sklearn's get_scorer.
        If callable, function with signature ``score_func(y, y_pred,
        **kwargs)``.

    Returns
    -------
    scorer : callable | None
        The scorer.
    """
    from sklearn.metrics import make_scorer, get_scorer
    # If scoring is None (default), the predictions are internally
    # generated by estimator.score(). Else, we must first get the
    # predictions based on the scorer.
    if scoring is None:
        return None
    elif isinstance(scoring, str):
        return get_scorer(scoring)
    else:
        return make_scorer(scoring)


def _get_inverse_funcs(estimator, terminal=True):
    """Retrieve the inverse functions of an pipeline or an estimator."""
    inverse_func = [False]
    if hasattr(estimator, 'steps'):
        # if pipeline, retrieve all steps by nesting
        inverse_func = list()
        for _, est in estimator.steps:
            inverse_func.extend(_get_inverse_funcs(est, terminal=False))
    elif hasattr(estimator, 'inverse_transform'):
        # if not pipeline attempt to retrieve inverse function
        inverse_func = [estimator.inverse_transform]

    # If terminal node, check that that the last estimator is a classifier,
    # and remove it from the transformers.
    if terminal:
        last_is_estimator = inverse_func[-1] is False
        all_invertible = not(False in inverse_func[:-1])
        if last_is_estimator and all_invertible:
            # keep all inverse transformation and remove last estimation
            inverse_func = inverse_func[:-1]
        else:
            inverse_func = list()

    return inverse_func


def get_coef(estimator, attr='filters_', inverse_transform=False):
    """Retrieve the coefficients of an estimator ending with a Linear Model.

    This is typically useful to retrieve "spatial filters" or "spatial
    patterns" of decoding models [1]_.

    Parameters
    ----------
    estimator : object | None
        An estimator from scikit-learn.
    attr : str
        The name of the coefficient attribute to retrieve, typically
        ``'filters_'`` (default) or ``'patterns_'``.
    inverse_transform : bool
        If True, returns the coefficients after inverse transforming them with
        the transformer steps of the estimator.

    Returns
    -------
    coef : array
        The coefficients.

    References
    ----------
    .. [1] Haufe, S., Meinecke, F., Gorgen, K., Dahne, S., Haynes, J.-D.,
       Blankertz, B., & Biessmann, F. (2014). On the interpretation of weight
       vectors of linear models in multivariate neuroimaging. NeuroImage, 87,
       96-110. doi:10.1016/j.neuroimage.2013.10.067.
    """
    # If searchlight, loop across estimators
    if hasattr(estimator, 'estimators_'):
        coef = list()
        for est in estimator.estimators_:
            coef.append(get_coef(est, attr, inverse_transform))
        return np.array(coef)

    else:
        # Get the coefficients of the last estimator in case of nested pipeline
        est = estimator
        while hasattr(est, 'steps'):
            est = est.steps[-1][1]
        if not hasattr(est, attr):
            raise ValueError('This estimator does not have a %s '
                             'attribute.' % attr)
        coef = getattr(est, attr)

        # inverse pattern e.g. to get back physical units
        if inverse_transform:
            if not hasattr(estimator, 'steps'):
                raise ValueError('inverse_transform can only be applied onto '
                                 'pipeline estimators.')

            # The inverse_transform parameter will call this method on any
            # estimator contained in the pipeline, in reverse order.
            for inverse_func in _get_inverse_funcs(estimator)[::-1]:
                coef = inverse_func([coef])[0]
        return coef
