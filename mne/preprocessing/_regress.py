# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.pick import _picks_to_idx, pick_info
from .._fiff.proj import _needs_eeg_average_ref_proj
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..epochs import BaseEpochs
from ..evoked import Evoked
from ..io import BaseRaw
from ..utils import (
    _check_fname,
    _check_option,
    _check_preload,
    _import_h5io_funcs,
    _validate_type,
    copy_function_doc_to_method_doc_static,
    fill_doc_static,
    verbose_static,
)


@verbose_static("picks_good_data")
def regress_artifact(
    inst,
    picks=None,
    *,
    exclude="bads",
    picks_artifact="eog",
    betas=None,
    proj=True,
    copy=True,
    verbose=None,
):
    """Remove artifacts using regression based on reference channels.

    Parameters
    ----------
    inst : instance of Epochs | Raw
        The instance to process.
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick good data channels. Note
        that channels in ``info['bads']`` *will be included* if their names or
        indices are explicitly provided.
    exclude : list | 'bads'
        List of channels to exclude from the regression, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.

        .. versionadded:: 1.2
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").
    betas : ndarray, shape (n_picks, n_picks_ref) | None
        The regression coefficients to use. If None (default), they will be
        estimated from the data.
    proj : bool
        Whether to automatically apply SSP projection vectors before performing
        the regression. Default is ``True``.
    copy : bool
        If True (default), copy the instance before modifying it.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    inst : same type as the input data
        The processed data.
    betas : ndarray, shape (n_picks, n_picks_ref)
        The betas used during regression.

    Notes
    -----
    To implement the method outlined in :footcite:`GrattonEtAl1983`,
    remove the evoked response from epochs before estimating the
    regression coefficients, then apply those regression coefficients to the
    original data in two calls like (here for a single-condition ``epochs``
    only):

        >>> epochs_no_ave = epochs.copy().subtract_evoked()  # doctest:+SKIP
        >>> _, betas = mne.preprocessing.regress(epochs_no_ave)  # doctest:+SKIP
        >>> epochs_clean, _ = mne.preprocessing.regress(epochs, betas=betas)  # doctest:+SKIP

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    if betas is None:
        model = EOGRegression(
            picks=picks, exclude=exclude, picks_artifact=picks_artifact, proj=proj
        )
        model.fit(inst)
    else:
        # Create an EOGRegression object and load the given betas into it.
        picks = _picks_to_idx(inst.info, picks, exclude=exclude, none="data")
        picks_artifact = _picks_to_idx(inst.info, picks_artifact)
        want_betas_shape = (len(picks), len(picks_artifact))
        _check_option("betas.shape", betas.shape, (want_betas_shape,))
        model = EOGRegression(picks, picks_artifact, proj=proj)
        model.info_ = inst.info.copy()
        model.coef_ = betas
    return model.apply(inst, copy=copy), model.coef_


@fill_doc_static("picks_good_data")
class EOGRegression:
    """Remove EOG artifact signals from other channels by regression.

    Employs linear regression to remove signals captured by some channels,
    typically EOG, as described in :footcite:`GrattonEtAl1983`. You can also
    choose to fit the regression coefficients on evoked blink/saccade data and
    then apply them to continuous data, as described in
    :footcite:`CroftBarry2000`.

    Parameters
    ----------
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick good data channels. Note
        that channels in ``info['bads']`` *will be included* if their names or
        indices are explicitly provided.
    exclude : list | 'bads'
        List of channels to exclude from the regression, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    picks_artifact : array-like | str
        Channel picks to use as predictor/explanatory variables capturing
        the artifact of interest (default is "eog").
    proj : bool
        Whether to automatically apply SSP projection vectors before fitting
        and applying the regression. Default is ``True``.

    Attributes
    ----------
    coef_ : ndarray, shape (n, n)
        The regression coefficients. Only available after fitting.
    info_ : Info
        Channel information corresponding to the regression weights.
        Only available after fitting.
    picks : array-like | str
        Channels to perform the regression on.
    exclude : list | 'bads'
        Channels to exclude from the regression.
    picks_artifact : array-like | str
        The channels designated as containing the artifacts of interest.
    proj : bool
        Whether projections will be applied before performing the regression.

    Notes
    -----
    .. versionadded:: 1.2

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, picks=None, exclude="bads", picks_artifact="eog", proj=True):
        self.picks = picks
        self.exclude = exclude
        self.picks_artifact = picks_artifact
        self.proj = proj

    def fit(self, inst):
        """Fit EOG regression coefficients.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data on which the EOG regression weights should be fitted.

        Returns
        -------
        self : EOGRegression
            The fitted ``EOGRegression`` object. The regression coefficients
            are available as the ``.coef_`` and ``.intercept_`` attributes.

        Notes
        -----
        If your data contains EEG channels, make sure to apply the desired
        reference (see :func:`mne.set_eeg_reference`) before performing EOG
        regression.
        """
        picks, picks_artifact = self._check_inst(inst)

        # Calculate regression coefficients. Add a row of ones to also fit the
        # intercept.
        _check_preload(inst, "artifact regression")
        artifact_data = inst._data[..., picks_artifact, :]
        ref_data = artifact_data - np.mean(artifact_data, axis=-1, keepdims=True)
        if ref_data.ndim == 3:
            ref_data = ref_data.transpose(1, 0, 2)
            ref_data = ref_data.reshape(len(picks_artifact), -1)
        cov_ref = ref_data @ ref_data.T

        # Process each channel separately to reduce memory load
        coef = np.zeros((len(picks), len(picks_artifact)))
        for pi, pick in enumerate(picks):
            this_data = inst._data[..., pick, :]  # view
            # Subtract mean over time from every trial/channel
            cov_data = this_data - np.mean(this_data, -1, keepdims=True)
            cov_data = cov_data.reshape(1, -1)
            # Perform the linear regression
            coef[pi] = np.linalg.solve(cov_ref, ref_data @ cov_data.T).T[0]

        # Store relevant parameters in the object.
        self.coef_ = coef
        self.info_ = inst.info.copy()
        return self

    @fill_doc_static("copy_df")
    def apply(self, inst, copy=True):
        """Apply the regression coefficients to data.

        Parameters
        ----------
        inst : Raw | Epochs | Evoked
            The data on which to apply the regression.
        copy : bool
            If ``True``, data will be copied. Otherwise data may be modified in place.
            Defaults to ``True``.

        Returns
        -------
        inst : Raw | Epochs | Evoked
            A version of the data with the artifact channels regressed out.

        Notes
        -----
        Only works after ``.fit()`` has been used.

        References
        ----------
        .. footbibliography::
        """
        if copy:
            inst = inst.copy()
        picks, picks_artifact = self._check_inst(inst)

        # Check that the channels are compatible with the regression weights.
        ref_picks = _picks_to_idx(
            self.info_, self.picks, none="data", exclude=self.exclude
        )
        ref_picks_artifact = _picks_to_idx(self.info_, self.picks_artifact)
        if any(
            inst.ch_names[ch1] != self.info_["chs"][ch2]["ch_name"]
            for ch1, ch2 in zip(picks, ref_picks)
        ):
            raise ValueError(
                "Selected data channels are not compatible with "
                "the regression weights. Make sure that all data "
                "channels are present and in the correct order."
            )
        if any(
            inst.ch_names[ch1] != self.info_["chs"][ch2]["ch_name"]
            for ch1, ch2 in zip(picks_artifact, ref_picks_artifact)
        ):
            raise ValueError(
                "Selected artifact channels are not compatible "
                "with the regression weights. Make sure that all "
                "artifact channels are present and in the "
                "correct order."
            )

        _check_preload(inst, "artifact regression")
        artifact_data = inst._data[..., picks_artifact, :]
        ref_data = artifact_data - np.mean(artifact_data, -1, keepdims=True)
        for pi, pick in enumerate(picks):
            this_data = inst._data[..., pick, :]  # view
            this_data -= (self.coef_[pi] @ ref_data).reshape(this_data.shape)
        return inst

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_regression_weights")
    def plot(
        self,
        ch_type=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        mask_label_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        axes=None,
        colorbar=True,
        cbar_fmt="%1.1e",
        title=None,
        show=True,
    ):
        """Plot the regression weights of a fitted EOGRegression model.

        Parameters
        ----------
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the RMS for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        sensors : bool | str
            Whether to add markers for sensor locations. If :class:`str`, should be a
            valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
            Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
            default), black circles will be used.
        show_names : bool | callable
            If ``True``, show channel names next to each sensor marker. If callable,
            channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
            non-masked sensor names will be shown.
        mask : ndarray of bool, shape (n_channels,) | None
            Array indicating channel(s) to highlight with a distinct
            plotting style.
            Array elements set to ``True`` will be plotted
            with the parameters given in ``mask_params``. Defaults to ``None``,
            equivalent to an array of all ``False`` elements.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4)
        mask_label_params : dict | None
            Additional plotting parameters for significant sensor labels.
            Default (None) equals::

                dict(fontsize='medium', fontweight='bold')

            .. versionadded:: 1.13
        contours : int | array-like
            The number of contour lines to draw. If ``0``, no contours will be drawn.
            If a positive integer, that number of contour levels are chosen using the
            matplotlib tick locator (may sometimes be inaccurate, use array for
            accuracy). If array-like, the array values are used as the contour levels.
            The values should be in µV for EEG, fT for magnetometers and fT/m for
            gradiometers. If ``colorbar=True``, the colorbar will have ticks
            corresponding to the contour levels. Default is ``6``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will be
            drawn. If dict, each key refers to a tuple of x and y positions, the values
            in 'mask_pos' will serve as image mask.
            Alternatively, a matplotlib patch object can be passed for advanced
            masking options, either directly or as a function that returns patches
            (required for multi-axis plots). If None, nothing will be drawn.
            Defaults to 'head'.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        image_interp : str
            The image interpolation to be used. Options are ``'cubic'`` (default)
            to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
            ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
            ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
        extrapolate : str
            Options:

            - ``'box'``
                Extrapolate to four points placed to form a square encompassing all
                data points, where each side of the square is three times the range
                of the data in the respective dimension.
            - ``'local'`` (default for MEG sensors)
                Extrapolate only to nearby points (approximately to points closer than
                median inter-electrode distance). This will also set the
                mask to be polygonal based on the convex hull of the sensors.
            - ``'head'`` (default for non-MEG sensors)
                Extrapolate out to the edges of the clipping circle. This will be on
                the head circle when the sensors are contained within the head circle,
                but it can extend beyond the head when sensors are plotted outside
                the head circle.

            .. versionchanged:: 0.21

               - The default was changed to ``'local'`` for MEG sensors.
               - ``'local'`` was changed to use a convex hull mask
               - ``'head'`` was changed to extrapolate out to the clipping circle.
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.

            .. versionadded:: 0.20
        res : int
            The resolution of the topomap image (number of pixels along each side).
        size : float
            Side length of each subplot in inches.
        cmap : str | matplotlib.colors.Colormap | tuple | 'interactive' | None
            Colormap to use. If :class:`tuple`, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging the
            colorbar with left and right mouse button. Left mouse button moves the
            scale up and down and right mouse button adjusts the range. Hitting
            space bar resets the range. Up and down arrows can be used to change
            the colormap. If ``None``, ``'Reds'`` is used for data that is either
            all-positive or all-negative, and ``'RdBu_r'`` is used otherwise.
            ``'interactive'`` is equivalent to ``(None, True)``. Defaults to ``None``.

            .. warning::  Interactive mode works smoothly only for a small amount
                of topomaps. Interactive mode is disabled by default for more than
                2 topomaps.
        vlim : tuple of length 2
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data.
            If both entries are ``None``, the bounds are set at
            ``(min(data), max(data))``.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. Defaults to ``(None, None)``.
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of ``times`` provided (unless ``times`` is
            ``None``). Default is ``None``.
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
        title : str | None
            The title of the generated figure. If ``None`` (default), no title is
            displayed.
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure with a topomap subplot for each channel type.

        Notes
        -----
        .. versionadded:: 1.2
        """  # noqa: E501
        from ..viz import plot_regression_weights

        return plot_regression_weights(
            self,
            ch_type=ch_type,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            mask_label_params=mask_label_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            axes=axes,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            title=title,
            show=show,
        )

    def _check_inst(self, inst):
        """Perform some sanity checks on the input."""
        _validate_type(
            inst, (BaseRaw, BaseEpochs, Evoked), "inst", "Raw, Epochs, Evoked"
        )
        picks = _picks_to_idx(inst.info, self.picks, none="data", exclude=self.exclude)
        picks_artifact = _picks_to_idx(inst.info, self.picks_artifact)
        all_picks = np.unique(np.concatenate([picks, picks_artifact]))
        use_info = pick_info(inst.info, all_picks)
        del all_picks
        if _needs_eeg_average_ref_proj(use_info):
            raise RuntimeError(
                "No average reference for the EEG channels has been "
                "set. Use inst.set_eeg_reference(projection=True) to do so."
            )
        if self.proj and not inst.proj:
            inst.apply_proj()
        if not inst.proj and len(use_info.get("projs", [])) > 0:
            raise RuntimeError(
                "Projections need to be applied before "
                "regression can be performed. Use the "
                ".apply_proj() method to do so."
            )
        return picks, picks_artifact

    def __repr__(self):
        """Produce a string representation of this object."""
        s = "<EOGRegression | "
        if hasattr(self, "coef_"):
            n_art = self.coef_.shape[1]
            plural = "s" if n_art > 1 else ""
            s += f"fitted to {n_art} artifact channel{plural}>"
        else:
            s += "not fitted>"
        return s

    @fill_doc_static("overwrite")
    def save(self, fname, overwrite=False):
        """Save the regression model to an HDF5 file.

        Parameters
        ----------
        fname : path-like
            The file to write the regression weights to. Should end in ``.h5``.
        overwrite : bool
            If True (default False), overwrite the destination file if it
            exists.
        """
        _, write_hdf5 = _import_h5io_funcs()
        _validate_type(fname, "path-like", "fname")
        fname = _check_fname(fname, overwrite=overwrite, name="fname")
        write_hdf5(fname, self.__dict__, overwrite=overwrite)


def read_eog_regression(fname):
    """Read an EOG regression model from an HDF5 file.

    Parameters
    ----------
    fname : path-like
        The file to read the regression model from. Should end in ``.h5``.

    Returns
    -------
    model : EOGRegression
        The regression model read from the file.

    Notes
    -----
    .. versionadded:: 1.2
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, "path-like", "fname")
    fname = _check_fname(fname, overwrite="read", must_exist=True, name="fname")
    model = EOGRegression()
    model.__dict__.update(read_hdf5(fname))
    return model
