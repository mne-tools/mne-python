# -*- coding: utf-8 -*-
# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#          Clemens Brunner <clemens.brunner@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import copy as cp

import numpy as np
from scipy import linalg

from .mixin import TransformerMixin
from .base import BaseEstimator
from ..cov import _regularized_covariance
from ..utils import fill_doc, _check_option


@fill_doc
class CSP(TransformerMixin, BaseEstimator):
    u"""M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    CSP in the context of EEG was first described in [1]; a comprehensive
    tutorial on CSP can be found in [2]. Multiclass solving is implemented
    from [3].

    Parameters
    ----------
    n_components : int, default 4
        The number of components to decompose M/EEG signals.
        This number should be set by cross-validation.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If transform_into == 'average_power' and log is None or True, then
        applies a log transform to standardize the features, else the features
        are z-scored. If transform_into == 'csp_space', then log must be None.
    cov_est : 'concat' | 'epoch', default 'concat'
        If 'concat', covariance matrices are estimated on concatenated epochs
        for each class.
        If 'epoch', covariance matrices are estimated on each epoch separately
        and then averaged over each class.
    transform_into : {'average_power', 'csp_space'}
        If 'average_power' then self.transform will return the average power of
        each spatial filter. If 'csp_space' self.transform will return the data
        in CSP space. Defaults to 'average_power'.
    norm_trace : bool
        Normalize class covariance by its trace. Defaults to False. Trace
        normalization is a step of the original CSP algorithm [1]_ to eliminate
        magnitude variations in the EEG between individuals. It is not applied
        in more recent work [2]_, [3]_ and can have a negative impact on
        patterns ordering.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    %(rank_None)s

        .. versionadded:: 0.17

    Attributes
    ----------
    filters_ :  ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_components,)
        If fit, the std squared power for each component.

    See Also
    --------
    mne.preprocessing.Xdawn, SPoC

    References
    ----------
    .. [1] Zoltan J. Koles, Michael S. Lazar, Steven Z. Zhou. Spatial Patterns
           Underlying Population Differences in the Background EEG. Brain
           Topography 2(4), 275-284, 1990.
    .. [2] Benjamin Blankertz, Ryota Tomioka, Steven Lemm, Motoaki Kawanabe,
           Klaus-Robert Müller. Optimizing Spatial Filters for Robust EEG
           Single-Trial Analysis. IEEE Signal Processing Magazine 25(1), 41-56,
           2008.
    .. [3] Grosse-Wentrup, Moritz, and Martin Buss. Multiclass common spatial
           patterns and information theoretic feature extraction. IEEE
           Transactions on Biomedical Engineering, Vol 55, no. 8, 2008.
    """

    def __init__(self, n_components=4, reg=None, log=None, cov_est="concat",
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None):
        """Init of CSP."""
        # Init default CSP
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        self.rank = rank

        self.reg = reg

        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        # Init default transform_into
        _check_option('transform_into', transform_into,
                      ['average_power', 'csp_space'])
        self.transform_into = transform_into

        # Init default log
        if transform_into == 'average_power':
            if log is not None and not isinstance(log, bool):
                raise ValueError('log must be a boolean if transform_into == '
                                 '"average_power".')
        else:
            if log is not None:
                raise ValueError('log must be a None if transform_into == '
                                 '"csp_space".')
        self.log = log

        if not isinstance(norm_trace, bool):
            raise ValueError('norm_trace must be a bool.')
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params

    def _check_Xy(self, X, y=None):
        """Aux. function to check input data."""
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        self._check_Xy(X, y)
        n_channels = X.shape[1]

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = np.zeros((n_classes, n_channels, n_channels))
        sample_weights = list()
        for class_idx, this_class in enumerate(self._classes):
            if self.cov_est == "concat":  # concatenate epochs
                class_ = np.transpose(X[y == this_class], [1, 0, 2])
                class_ = class_.reshape(n_channels, -1)
                cov = _regularized_covariance(
                    class_, reg=self.reg, method_params=self.cov_method_params,
                    rank=self.rank)
                weight = sum(y == this_class)
            elif self.cov_est == "epoch":
                class_ = X[y == this_class]
                cov = np.zeros((n_channels, n_channels))
                for this_X in class_:
                    cov += _regularized_covariance(
                        this_X, reg=self.reg,
                        method_params=self.cov_method_params,
                        rank=self.rank)
                cov /= len(class_)
                weight = len(class_)

            covs[class_idx] = cov
            if self.norm_trace:
                # Append covariance matrix and weight. Prior to version 0.15,
                # trace normalization was applied, but was breaking results for
                # some usecases by changing the apparent ranking of patterns.
                # Trace normalization of the covariance matrix was removed
                # without signigificant effect on patterns or performances.
                # If the user interested in this feature, we suggest trace
                # normalization of the epochs prior to the CSP.
                covs[class_idx] /= np.trace(cov)

            sample_weights.append(weight)

        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
            # sort eigenvectors
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)

            # Here we apply an euclidean mean. See pyRiemann for other metrics
            mean_cov = np.average(covs, axis=0, weights=sample_weights)
            eigen_vectors = eigen_vectors.T

            # normalize
            for ii in range(eigen_vectors.shape[1]):
                tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov),
                             eigen_vectors[:, ii])
                eigen_vectors[:, ii] /= np.sqrt(tmp)

            # class probability
            class_probas = [np.mean(y == _class) for _class in self._classes]

            # mutual information
            mutual_info = []
            for jj in range(eigen_vectors.shape[1]):
                aa, bb = 0, 0
                for (cov, prob) in zip(covs, class_probas):
                    tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov),
                                 eigen_vectors[:, jj])
                    aa += prob * np.log(np.sqrt(tmp))
                    bb += prob * (tmp ** 2 - 1)
                mi = - (aa + (3.0 / 16) * (bb ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]

        # sort eigenvectors
        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = linalg.pinv2(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the CSP filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_sources, n_times).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X

    @fill_doc
    def plot_patterns(self, info, components=None, ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scalings=None, units='a.u.', res=64,
                      size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                      show=True, show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None,
                      sphere=None):
        """Plot topographic patterns of components.

        The patterns explain how the measured data was generated from the
        neural sources (a.k.a. the forward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used for fitting.
            If not possible, consider using ``create_info``.
        components : float | array of float | None
           The patterns to plot. If None, n_components will be shown.
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
            The value specifying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specifying the upper bound of the color range.
            If None, the maximum absolute value is used. If vmin is None,
            but vmax is not, default np.min(data).
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
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        units : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        name_format : str
            String format for topomap values. Defaults to "CSP%%01d".
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
        %(topomap_outlines)s
        contours : int | array of float
            The number of contour lines to draw. If 0, no contours will be
            drawn. When an integer, matplotlib ticker locator is used to find
            suitable values for the contour thresholds (may sometimes be
            inaccurate, use array for accuracy). If an array, the values
            represent the levels for the contours. Defaults to 6.
        image_interp : str
            The image interpolation to be used.
            All matplotlib options are accepted.
        average : float | None
            The time window around a given time to be used for averaging
            (seconds). For example, 0.01 would translate into window that
            starts 5 ms before and ends 5 ms after a given time point.
            Defaults to None, which means no averaging.
        %(topomap_head_pos)s
        %(topomap_sphere_auto)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        from .. import EvokedArray
        if components is None:
            components = np.arange(self.n_components)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        # create an evoked
        patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        # the call plot_topomap
        return patterns.plot_topomap(
            times=components, ch_type=ch_type, layout=layout,
            vmin=vmin, vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, head_pos=head_pos, sphere=sphere)

    @fill_doc
    def plot_filters(self, info, components=None, ch_type=None, layout=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic filters of components.

        The filters are used to extract discriminant neural sources from
        the measured data (a.k.a. the backward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used for fitting.
            If not possible, consider using ``create_info``.
        components : float | array of float | None
           The patterns to plot. If None, n_components will be shown.
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
            The value specifying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specifying the upper bound of the color range.
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
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        units : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        name_format : str
            String format for topomap values. Defaults to "CSP%%01d".
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
        %(topomap_outlines)s
        contours : int | array of float
            The number of contour lines to draw. If 0, no contours will be
            drawn. When an integer, matplotlib ticker locator is used to find
            suitable values for the contour thresholds (may sometimes be
            inaccurate, use array for accuracy). If an array, the values
            represent the levels for the contours. Defaults to 6.
        image_interp : str
            The image interpolation to be used.
            All matplotlib options are accepted.
        average : float | None
            The time window around a given time to be used for averaging
            (seconds). For example, 0.01 would translate into window that
            starts 5 ms before and ends 5 ms after a given time point.
            Defaults to None, which means no averaging.
        %(topomap_head_pos)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        from .. import EvokedArray
        if components is None:
            components = np.arange(self.n_components)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        # create an evoked
        filters = EvokedArray(self.filters_, info, tmin=0)
        # the call plot_topomap
        return filters.plot_topomap(
            times=components, ch_type=ch_type, layout=layout, vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, title=title, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average,
            head_pos=head_pos)


def _ajd_pham(X, eps=1e-6, max_iter=15):
    """Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    eps : float, default 1e-6
        The tolerance for stopping criterion.
    max_iter : int, default 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    n_epochs = X.shape[0]

    # Reshape input matrix
    A = np.concatenate(X, axis=0).T

    # Init variables
    n_times, n_m = A.shape
    V = np.eye(n_times)
    epsilon = n_times * (n_times - 1) * eps

    for it in range(max_iter):
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]

                g12 = np.mean(A[ii, Ij] / c1)
                g21 = np.mean(A[ii, Ij] / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if decr < epsilon:
            break
    D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
    return V, D


@fill_doc
class SPoC(CSP):
    """Implementation of the SPoC spatial filtering.

    Source Power Comodulation (SPoC) [1]_ allows to extract spatial filters and
    patterns by using a target (continuous) variable in the decomposition
    process in order to give preference to components whose power correlates
    with the target variable.

    SPoC can be seen as an extension of the CSP driven by a continuous
    variable rather than a discrete variable. Typical applications include
    extraction of motor patterns using EMG power or audio patterns using sound
    envelope.

    Parameters
    ----------
    n_components : int
        The number of components to decompose M/EEG signals.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow
        regularization for covariance estimation.
        If float, shrinkage is used (0 <= shrinkage <= 1).
        For str options, ``reg`` will be passed to ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If transform_into == 'average_power' and log is None or True, then
        applies a log transform to standardize the features, else the features
        are z-scored. If transform_into == 'csp_space', then log must be None.
    transform_into : {'average_power', 'csp_space'}
        If 'average_power' then self.transform will return the average power of
        each spatial filter. If 'csp_space' self.transform will return the data
        in CSP space. Defaults to 'average_power'.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    %(rank_None)s

        .. versionadded:: 0.17

    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_channels)
        If fit, the SPoC spatial filters, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the SPoC spatial patterns, else None.
    mean_ : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_components,)
        If fit, the std squared power for each component.

    See Also
    --------
    mne.preprocessing.Xdawn, CSP

    References
    ----------
    .. [1] Dahne, S., Meinecke, F. C., Haufe, S., Hohne, J., Tangermann, M.,
           Muller, K. R., & Nikulin, V. V. (2014). SPoC: a novel framework for
           relating the amplitude of neuronal oscillations to behaviorally
           relevant parameters. NeuroImage, 86, 111-122.
    """

    def __init__(self, n_components=4, reg=None, log=None,
                 transform_into='average_power', cov_method_params=None,
                 rank=None):
        """Init of SPoC."""
        super(SPoC, self).__init__(n_components=n_components, reg=reg, log=log,
                                   cov_est="epoch", norm_trace=False,
                                   transform_into=transform_into, rank=rank,
                                   cov_method_params=cov_method_params)
        # Covariance estimation have to be done on the single epoch level,
        # unlike CSP where covariance estimation can also be achieved through
        # concatenation of all epochs from the same class.
        delattr(self, 'cov_est')
        delattr(self, 'norm_trace')

    def fit(self, X, y):
        """Estimate the SPoC decomposition on epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the SPoC.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of SPoC
            Returns the modified instance.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        self._check_Xy(X, y)

        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")

        # The following code is directly copied from pyRiemann

        # Normalize target variable
        target = y.astype(np.float64)
        target -= target.mean()
        target /= target.std()

        n_epochs, n_channels = X.shape[:2]

        # Estimate single trial covariance
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X):
            covs[ii] = _regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)

        C = covs.mean(0)
        Cz = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)

        # solve eigenvalue decomposition
        evals, evecs = linalg.eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = linalg.pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the SPoC filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)
            If self.transform_into == 'csp_space' then returns the data in CSP
            space and shape is (n_epochs, n_sources, n_times).
        """
        return super(SPoC, self).transform(X)
