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

from .base import BaseEstimator
from .mixin import TransformerMixin
from ..cov import _regularized_covariance
from ..utils import fill_doc, _check_option, _validate_type


@fill_doc
class CSP(TransformerMixin, BaseEstimator):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    This class can be used as a supervised decomposition to estimate spatial
    filters for feature extraction. CSP in the context of EEG was first
    described in :footcite:`KolesEtAl1990`; a comprehensive tutorial on CSP can
    be found in :footcite:`BlankertzEtAl2008`. Multi-class solving is
    implemented from :footcite:`Grosse-WentrupBuss2008`.

    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals. This number should
        be set by cross-validation.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow regularization
        for covariance estimation. If float (between 0 and 1), shrinkage is
        used. For str values, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If ``transform_into`` equals ``'average_power'`` and ``log`` is None or
        True, then apply a log transform to standardize features, else features
        are z-scored. If ``transform_into`` is ``'csp_space'``, ``log`` must be
        None.
    cov_est : 'concat' | 'epoch' (default 'concat')
        If ``'concat'``, covariance matrices are estimated on concatenated
        epochs for each class. If ``'epoch'``, covariance matrices are
        estimated on each epoch separately and then averaged over each class.
    transform_into : 'average_power' | 'csp_space' (default 'average_power')
        If 'average_power' then ``self.transform`` will return the average
        power of each spatial filter. If ``'csp_space'``, ``self.transform``
        will return the data in CSP space.
    norm_trace : bool (default False)
        Normalize class covariance by its trace. Trace normalization is a step
        of the original CSP algorithm :footcite:`KolesEtAl1990` to eliminate
        magnitude variations in the EEG between individuals. It is not applied
        in more recent work :footcite:`BlankertzEtAl2008`,
        :footcite:`Grosse-WentrupBuss2008` and can have a negative impact on
        pattern order.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.

        .. versionadded:: 0.16
    %(rank_None)s

        .. versionadded:: 0.17
    component_order : 'mutual_info' | 'alternate' (default 'mutual_info')
        If ``'mutual_info'`` order components by decreasing mutual information
        (in the two-class case this uses a simplification which orders
        components by decreasing absolute deviation of the eigenvalues from 0.5
        :footcite:`BarachantEtAl2010`). For the two-class case, ``'alternate'``
        orders components by starting with the largest eigenvalue, followed by
        the smallest, the second-to-largest, the second-to-smallest, and so on
        :footcite:`BlankertzEtAl2008`.

        .. versionadded:: 0.21

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
    .. footbibliography::
    """

    def __init__(self, n_components=4, reg=None, log=None, cov_est='concat',
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info'):
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
        self.transform_into = _check_option('transform_into', transform_into,
                                            ['average_power', 'csp_space'])

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

        _validate_type(norm_trace, bool, 'norm_trace')
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.component_order = _check_option('component_order',
                                             component_order,
                                             ('mutual_info', 'alternate'))

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
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
        self._check_Xy(X, y)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        if n_classes > 2 and self.component_order == 'alternate':
            raise ValueError("component_order='alternate' requires two "
                             "classes, but data contains {} classes; use "
                             "component_order='mutual_info' "
                             "instead.".format(n_classes))

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs,
                                                           sample_weights)
        ix = self._order_components(covs, sample_weights, eigen_vectors,
                                    eigen_values, self.component_order)

        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = linalg.pinv2(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
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
    def plot_patterns(self, info, components=None, ch_type=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scalings=None, units='a.u.', res=64,
                      size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                      show=True, show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None,
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
            function lambda x: x.replace('MEG ', ''). If ``mask`` is not None,
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
            times=components, ch_type=ch_type,
            vmin=vmin, vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, sphere=sphere)

    @fill_doc
    def plot_filters(self, info, components=None, ch_type=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None):
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
            function lambda x: x.replace('MEG ', ''). If ``mask`` is not None,
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
        filters = EvokedArray(self.filters_.T, info, tmin=0)
        # the call plot_topomap
        return filters.plot_topomap(
            times=components, ch_type=ch_type, vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, title=title, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average)

    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        if self.cov_est == "concat":
            cov_estimator = self._concat_cov
        elif self.cov_est == "epoch":
            cov_estimator = self._epoch_cov

        covs = []
        sample_weights = []
        for this_class in self._classes:
            cov, weight = cov_estimator(X[y == this_class])

            if self.norm_trace:
                cov /= np.trace(cov)

            covs.append(cov)
            sample_weights.append(weight)

        return np.stack(covs), np.array(sample_weights)

    def _concat_cov(self, x_class):
        """Concatenate epochs before computing the covariance."""
        _, n_channels, _ = x_class.shape

        x_class = np.transpose(x_class, [1, 0, 2])
        x_class = x_class.reshape(n_channels, -1)
        cov = _regularized_covariance(
            x_class, reg=self.reg, method_params=self.cov_method_params,
            rank=self.rank)
        weight = x_class.shape[0]

        return cov, weight

    def _epoch_cov(self, x_class):
        """Mean of per-epoch covariances."""
        cov = sum(_regularized_covariance(
            this_X, reg=self.reg,
            method_params=self.cov_method_params,
            rank=self.rank) for this_X in x_class)
        cov /= len(x_class)
        weight = len(x_class)

        return cov, weight

    def _decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = _ajd_pham(covs)
            eigen_vectors = self._normalize_eigenvectors(eigen_vectors.T, covs,
                                                         sample_weights)
            eigen_values = None
        return eigen_vectors, eigen_values

    def _compute_mutual_info(self, covs, sample_weights, eigen_vectors):
        class_probas = sample_weights / sample_weights.sum()

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

        return mutual_info

    def _normalize_eigenvectors(self, eigen_vectors, covs, sample_weights):
        # Here we apply an euclidean mean. See pyRiemann for other metrics
        mean_cov = np.average(covs, axis=0, weights=sample_weights)

        for ii in range(eigen_vectors.shape[1]):
            tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov),
                         eigen_vectors[:, ii])
            eigen_vectors[:, ii] /= np.sqrt(tmp)
        return eigen_vectors

    def _order_components(self, covs, sample_weights, eigen_vectors,
                          eigen_values, component_order):
        n_classes = len(self._classes)
        if component_order == 'mutual_info' and n_classes > 2:
            mutual_info = self._compute_mutual_info(covs, sample_weights,
                                                    eigen_vectors)
            ix = np.argsort(mutual_info)[::-1]
        elif component_order == 'mutual_info' and n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        elif component_order == 'alternate' and n_classes == 2:
            i = np.argsort(eigen_values)
            ix = np.empty_like(i)
            ix[1::2] = i[:len(i) // 2]
            ix[0::2] = i[len(i) // 2:][::-1]
        return ix


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

    Source Power Comodulation (SPoC) :footcite:`DahneEtAl2014` allows to
    extract spatial filters and
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
    .. footbibliography::
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
