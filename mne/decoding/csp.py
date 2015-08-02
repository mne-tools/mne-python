# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import copy as cp
import warnings

import numpy as np
from scipy import linalg

from .mixin import TransformerMixin
from ..cov import _regularized_covariance


class CSP(TransformerMixin):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).

    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    See [1].

    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals.
        This number should be set by cross-validation.
    reg : float | str | None (default None)
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').
    log : bool (default True)
        If true, apply log to standardize the features.
        If false, features are just z-scored.

    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_channels,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_channels,)
        If fit, the std squared power for each component.

    References
    ----------
    [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.
    """

    def __init__(self, n_components=4, reg=None, log=True):
        """Init of CSP."""
        self.n_components = n_components
        if reg == 'lws':
            warnings.warn('`lws` has been deprecated for the `reg`'
                          ' argument. It will be removed in 0.11.'
                          ' Use `ledoit_wolf` instead.', DeprecationWarning)
            reg = 'ledoit_wolf'
        self.reg = reg
        self.log = log
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, epochs_data, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        epochs_data : ndarray, shape (n_epochs, n_channels, n_times)
            The data to estimate the CSP on.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)
        # check number of epochs
        if epochs_data.shape[0] != len(y):
            raise ValueError("n_epochs must be the same for epochs_data and y")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")
        # concatenate epochs
        class_1 = np.transpose(epochs_data[y == classes[0]],
                               [1, 0, 2]).reshape(epochs_data.shape[1], -1)
        class_2 = np.transpose(epochs_data[y == classes[1]],
                               [1, 0, 2]).reshape(epochs_data.shape[1], -1)

        cov_1 = _regularized_covariance(class_1, reg=self.reg)
        cov_2 = _regularized_covariance(class_2, reg=self.reg)

        # then fit on covariance
        self._fit(cov_1, cov_2)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, e) for e in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def _fit(self, cov_a, cov_b):
        """Aux Function (modifies cov_a and cov_b in-place)."""
        cov_a /= np.trace(cov_a)
        cov_b /= np.trace(cov_b)
        # computes the eigen values
        lambda_, u = linalg.eigh(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = lambda_[ind]

        u = u[:, ind]
        p = np.dot(np.sqrt(linalg.pinv(np.diag(lambda2_))), u.T)

        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        vals, vecs = linalg.eigh(w_a, w_b)
        # sort vectors by discriminative power using eigen values
        ind = np.argsort(np.maximum(vals, 1. / vals))[::-1]
        vecs = vecs[:, ind]
        # and project
        w = np.dot(vecs.T, p)

        self.filters_ = w
        self.patterns_ = linalg.pinv(w).T

    def transform(self, epochs_data, y=None):
        """Estimate epochs sources given the CSP filters.

        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None
            Not used.

        Returns
        -------
        X : ndarray of shape (n_epochs, n_sources)
            The CSP features averaged over time.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, e) for e in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)
        if self.log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X

    def plot_patterns(self, info, components=None, ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scale=None, scale_time=1, unit=None,
                      res=64, size=1, cbar_fmt='%3.1f',
                      name_format='CSP%01d', proj=False, show=True,
                      show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic patterns of CSP components.

        The CSP patterns explain how the measured data was generated
        from the neural sources (a.k.a. the forward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used to fit CSP.
            If not possible, consider using ``create_info``.
        components : float | array of floats | None.
           The CSP patterns to plot. If None, n_components will be shown.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are
            collected in pairs and the RMS for each pair is plotted.
            If None, then channels are chosen in the order given above.
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
        cmap : matplotlib colormap
            Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
            'Reds'.
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
            Scale the time labels. Defaults to 1.
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
            String format for topomap values. Defaults to "CSP%01d"
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
            Indicies set to `True` will be considered. Defaults to None.
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
        if components is None:
            components = np.arange(self.n_components)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        # create an evoked
        patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        # the call plot_topomap
        return patterns.plot_topomap(times=components, ch_type=ch_type,
                                     layout=layout, vmin=vmin, vmax=vmax,
                                     cmap=cmap, colorbar=colorbar, res=res,
                                     cbar_fmt=cbar_fmt, sensors=sensors,
                                     scale=1, scale_time=1, unit='a.u.',
                                     time_format=name_format, size=size,
                                     show_names=show_names,
                                     mask_params=mask_params,
                                     mask=mask, outlines=outlines,
                                     contours=contours,
                                     image_interp=image_interp, show=show,
                                     head_pos=head_pos)

    def plot_filters(self, info, components=None, ch_type=None, layout=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scale=None, scale_time=1, unit=None,
                     res=64, size=1, cbar_fmt='%3.1f',
                     name_format='CSP%01d', proj=False, show=True,
                     show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic filters of CSP components.

        The CSP filters are used to extract discriminant neural sources from
        the measured data (a.k.a. the backward model).

        Parameters
        ----------
        info : instance of Info
            Info dictionary of the epochs used to fit CSP.
            If not possible, consider using ``create_info``.
        components : float | array of floats | None.
           The CSP patterns to plot. If None, n_components will be shown.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are
            collected in pairs and the RMS for each pair is plotted.
            If None, then channels are chosen in the order given above.
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
        cmap : matplotlib colormap
            Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
            'Reds'.
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
            Scale the time labels. Defaults to 1.
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
            String format for topomap values. Defaults to "CSP%01d"
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
            Indicies set to `True` will be considered. Defaults to None.
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
        if components is None:
            components = np.arange(self.n_components)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        # create an evoked
        filters = EvokedArray(self.filters_, info, tmin=0)
        # the call plot_topomap
        return filters.plot_topomap(times=components, ch_type=ch_type,
                                    layout=layout, vmin=vmin, vmax=vmax,
                                    cmap=cmap, colorbar=colorbar, res=res,
                                    cbar_fmt=cbar_fmt, sensors=sensors,
                                    scale=1, scale_time=1, unit='a.u.',
                                    time_format=name_format, size=size,
                                    show_names=show_names,
                                    mask_params=mask_params,
                                    mask=mask, outlines=outlines,
                                    contours=contours,
                                    image_interp=image_interp, show=show,
                                    head_pos=head_pos)
