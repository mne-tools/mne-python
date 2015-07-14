# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np


class LinearClassifier():
    """
    This object clones a Linear Classifier from sklearn
    and updates the attribute for each fit. The model coefficient
    can be interpreted using the attribute patterns [1].

    Parameters
    ----------
    clf : object | None
        A linear classifier from sklearn with a fit method 
        that updates a coef_ attribute.
        If None the classifier will be a LogisticRegression

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filters used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the patterns used to restore M/EEG signals, else None.
    
    References
    ----------
    """
    def __init__(self, clf=None):
        if clf is None:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        
        self.clf = clf
    
    def fit(self, X, y):
        """Estimate the coeffiscient of the linear classifier.
        Save the coeffiscient in the attribute filters_ and 
        computes the attribute patterns_ using [1].

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to estimate the coeffiscient.
        y : array, shape=(n_epochs)
            The class for each epoch.

        Returns
        -------
        self : instance of LinearClassifier
            Returns the modified instance.
        
        References
        ----------
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_epochs x n_features)"
                             " instead of ", X.shape)
        # check for number of classes
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least two different classes in the data.")
        
        # fit the classifier
        self.clf.fit(X, y)
        # computes the patterns
        if hasattr(self.clf, 'coef_'):
            self.patterns_ = np.dot(X.T, np.dot(X, self.clf.coef_.T))
            self.filters_ = self.clf.coef_
        
        return self
    
    def transform(self, X, y=None):
        """Transform the data using the linear classifier.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to transform.
        y : array, shape=(n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape=(n_epochs)
            Predicted class label per epoch.
        
        """
        return self.predict(X)
    
    
    def fit_transform(self, X, y):
        """fit the data and transform it using the linear classifier.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to transform.
        y : array, shape=(n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape=(n_epochs)
            Predicted class label per epoch.
        
        """
        return self.fit(X, y).predict(X)
    
    def predict(self, X):
        """Predict class labels for each trial in X.
        
        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The features of each trial to predict class label.
        
        Returns
        -------
        y_pred : array, shape=(n_epochs)
            Predicted class label per epoch.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_epochs x n_features)"
                             " instead of ", X.shape)
        
        y_pred = self.clf.predict(X)
        return y_pred
    
    def plot_patterns(self, info, times=None, ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scale=None, scale_time=1e3, unit=None,
                      res=64, size=1, cbar_fmt='%3.1f',
                      time_format='%01d ms', proj=False, show=True,
                      show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None):
        """Plot topographic patterns of the linear classifier
        Parameters
        ----------
        self : instance of self
            a linear classifier 
        info : instance of Info
            Info dictionary of the epochs used to fit the linear classifier.
            If not possible, consider using ``create_info``.
        times : float | array of floats | None.
            The time point(s) to plot. If None, 10 topographies will be shown
            will a regular time spacing between the first and last time instant.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are collected in
            pairs and the RMS for each pair is plotted.
            If None, then channels are chosen in the order given above.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to
            be specified for Neuromag data). If possible, the correct layout file
            is inferred from the data; if no appropriate layout file was found, the
            layout is automatically generated from the sensor locations.
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
            Add markers for sensor locations to the plot. Accepts matplotlib plot
            format string (e.g., 'r+' for red plusses). If True, a circle will be
            used (via .add_artist). Defaults to True.
        colorbar : bool
            Plot a colorbar.
        scale : dict | float | None
            Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
            for grad and 1e15 for mag.
        scale_time : float | None
            Scale the time labels. Defaults to 1e3 (ms).
        unit : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        time_format : str
            String format for topomap values. Defaults to "%01d ms"
        proj : bool | 'interactive'
            If true SSP projections are applied before display. If 'interactive',
            a check box for reversible selection of SSP projection vectors will
            be show.
        show : bool
            Show figure if True.
        show_names : bool | callable
            If True, show channel names on top of the map. If a callable is
            passed, channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            lambda x: x.replace('MEG ', ''). If `mask` is not None, only
            significant sensors will be shown.
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

        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', a head scheme will be drawn. If
            dict, each key refers to a tuple of x and y positions. The values in
            'mask_pos' will serve as image mask. If None, nothing will be drawn.
            Defaults to 'head'. If dict, the 'autoshrink' (bool) field will
            trigger automated shrinking of the positions due to points outside the
            outline. Moreover, a matplotlib patch object can be passed for
            advanced masking options, either directly or as a function that returns
            patches (required for multi-axis plots).
        contours : int | False | None
            The number of contour lines to draw. If 0, no contours will be drawn.
        image_interp : str
            The image interpolation to be used. All matplotlib options are
            accepted.
        average : float | None
            The time window around a given time to be used for averaging (seconds).
            For example, 0.01 would translate into window that starts 5 ms before
            and ends 5 ms after a given time point. Defaults to None, which means
            no averaging.
        head_pos : dict | None
            If None (default), the sensors are positioned such that they span
            the head circle. If dict, can have entries 'center' (tuple) and
            'scale' (tuple) for what the center and scale of the head should be
            relative to the electrode locations.
        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        from mne import EvokedArray
        # create an evoked 
        pat = self.patterns_.reshape(info['nchan'], -1)
        patterns = EvokedArray(pat, info, tmin=0)
        # the call plot_topomap
        return patterns.plot_topomap(times=times, ch_type=ch_type, layout=layout,
                                     vmin=vmin, vmax=vmax, cmap=cmap, 
                                     colorbar=colorbar, res=res, cbar_fmt=cbar_fmt, 
                                     sensors=sensors, scale=scale, scale_time=scale_time, 
                                     unit=unit, time_format=time_format, size=size, 
                                     show_names=show_names, mask_params=mask_params, 
                                     mask=mask, outlines=outlines, contours=contours, 
                                     image_interp=image_interp, show=False)
        
    
        