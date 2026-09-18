"""Container classes for spectral data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy
from functools import partial
from inspect import signature

import numpy as np

from .._fiff.meas_info import ContainsMixin, Info
from .._fiff.pick import _pick_data_channels, _picks_to_idx, pick_info
from ..channels.channels import UpdateChannelsMixin
from ..channels.layout import _merge_ch_data, find_layout
from ..defaults import (
    _BORDER_DEFAULT,
    _EXTRAPOLATE_DEFAULT,
    _INTERPOLATION_DEFAULT,
    _handle_default,
)
from ..html_templates import _get_html_template
from ..utils import (
    GetEpochsMixin,
    _build_data_frame,
    _check_method_kwargs,
    _check_pandas_index_arguments,
    _check_pandas_installed,
    _check_sphere,
    _time_mask,
    _validate_type,
    _verbose_control,
    fill_doc_static,
    legacy,
    logger,
    object_diff,
    repr_html,
    verbose_static,
    warn,
)
from ..utils.check import (
    _check_fname,
    _check_option,
    _import_h5io_funcs,
    _is_numeric,
    check_fname,
)
from ..utils.misc import _pl
from ..utils.spectrum import (
    _convert_old_birthday_format,
    _get_instance_type_string,
    _split_psd_kwargs,
)
from .multitaper import _psd_from_mt, psd_array_multitaper
from .psd import _check_nfft, psd_array_welch


class SpectrumMixin:
    """Mixin providing spectral plotting methods to sensor-space containers."""

    @legacy(alt=".compute_psd().plot()")
    @verbose_static(
        "fmin_fmax_psd",
        "tmin_tmax_psd",
        "picks_good_data_noref",
        "proj_psd",
        "reject_by_annotation_psd",
        "method_plot_psd_auto",
        "average_plot_psd",
        "dB_plot_psd",
        "estimate_plot_psd",
        "xscale_plot_psd",
        "area_mode_plot_psd",
        "area_alpha_plot_psd",
        "color_plot_psd",
        "line_alpha_plot_psd",
        "spatial_colors_psd",
        "sphere_topomap_auto",
        "ax_plot_psd",
        "show",
        "n_jobs",
        "method_kw_psd",
        "notes_plot_psd_meth",
    )
    def plot_psd(
        self,
        fmin=0,
        fmax=np.inf,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        reject_by_annotation=True,
        *,
        method="auto",
        average=False,
        dB=True,
        estimate="power",
        xscale="linear",
        area_mode="std",
        area_alpha=0.33,
        color="black",
        line_alpha=None,
        spatial_colors=True,
        sphere=None,
        exclude="bads",
        ax=None,
        show=True,
        n_jobs=1,
        verbose=None,
        **method_kw,
    ):
        """Plot power or amplitude spectra.

        Separate plots are drawn for each channel type. When the data have been
        processed with a bandpass, lowpass or highpass filter, dashed lines (╎)
        indicate the boundaries of the filter. The line noise frequency is also
        indicated with a dashed line (⋮). If ``average=False``, the plot will
        be interactive, and click-dragging on the spectrum will generate a
        scalp topography plot for the chosen frequency range in a new figure.

        Parameters
        ----------
        fmin, fmax : float
            The lower- and upper-bound on frequencies of interest. Default is
            ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        reject_by_annotation : bool
            Whether to omit bad spans of data before spectral estimation. If
            ``True``, spans with annotations whose description begins with
            ``bad`` will be omitted.
        method : ``'welch'`` | ``'multitaper'`` | ``'auto'``
            Spectral estimation method. ``'welch'`` uses Welch's
            method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
            tapers :footcite:p:`Slepian1978`. ``'auto'`` (default) uses Welch's
            method for continuous data and multitaper for
            :class:`~mne.Epochs` or :class:`~mne.Evoked` data.
        average : bool
            If False, the PSDs of all channels is displayed. No averaging
            is done and parameters area_mode and area_alpha are ignored. When
            False, it is possible to paint an area (hold left mouse button and
            drag) to plot a topomap.
        dB : bool
            Plot power spectral density (PSD) in units (dB/Hz) if ``dB=True`` and
            ``estimate='power'``. Plot PSD in units (amplitude**2/Hz) if ``dB=False``
            and ``estimate='power'``. Plot amplitude spectral density (ASD) in units
            (amplitude/sqrt(Hz)) if ``dB=False`` and ``estimate='amplitude'``. Plot ASD
            in units (dB/sqrt(Hz)) if ``dB=True`` and ``estimate='amplitude'``.
        estimate : str, {'power', 'amplitude'}
            Can be "power" for power spectral density (PSD; default), "amplitude" for
            amplitude spectrum density (ASD).
        xscale : 'linear' | 'log'
            Scale of the frequency axis. Default is ``'linear'``.
        area_mode : str | None
            Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
            will be plotted. If 'range', the min and max (across channels) will be
            plotted. Bad channels will be excluded from these calculations.
            If None, no area will be plotted. If average=False, no area is plotted.
        area_alpha : float
            Alpha for the area.
        color : str | tuple
            A matplotlib-compatible color to use. Has no effect when
            spatial_colors=True.
        line_alpha : float | None
            Alpha for the PSD line. Can be None (default) to use 1.0 when
            ``average=True`` and 0.1 when ``average=False``.
        spatial_colors : bool
            Whether to color spectrum lines by channel location. Ignored if
            ``average=True``.
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

            .. versionadded:: 0.22.0
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the bad
            channels are excluded. Pass an empty list to plot all channels
            (including channels marked "bad", if any).

            .. versionadded:: 0.24.0
        ax : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of channel types present in
            the object. Default is ``None``.
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
        **method_kw
            Additional keyword arguments passed to the spectral estimation
            function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
            for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
            for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
            and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
            that for Welch method if ``n_fft`` is unspecified its default will be
            the smaller of ``2048`` or the number of available time samples (taking into
            account ``tmin`` and ``tmax``), not ``256`` as in
            :func:`~mne.time_frequency.psd_array_welch`.

        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.

        Notes
        -----
        This method exists to support legacy code; for new code the preferred
        idiom is ``inst.compute_psd().plot()`` (where ``inst`` is an instance
        of :class:`~mne.io.Raw`, :class:`~mne.Epochs`, or :class:`~mne.Evoked`).
        """  # noqa: E501
        init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot)
        return self.compute_psd(**init_kw).plot(**plot_kw)

    @legacy(alt=".compute_psd().plot_topo()")
    @verbose_static(
        "tmin_tmax_psd",
        "fmin_fmax_psd_topo",
        "proj_psd",
        "method_plot_psd_auto",
        "dB_spectrum_plot_topo",
        "layout_spectrum_plot_topo",
        "color_spectrum_plot_topo",
        "fig_facecolor",
        "axis_facecolor",
        "axes_spectrum_plot_topo",
        "show",
        "n_jobs",
        "method_kw_psd",
    )
    def plot_psd_topo(
        self,
        tmin=None,
        tmax=None,
        fmin=0,
        fmax=100,
        proj=False,
        *,
        method="auto",
        dB=True,
        layout=None,
        color="w",
        fig_facecolor="k",
        axis_facecolor="k",
        axes=None,
        show=True,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        """Plot power spectral density, separately for each channel.

        Parameters
        ----------
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        fmin, fmax : float
            The lower- and upper-bound on frequencies of interest. Default is
            ``fmin=0, fmax=100``.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        method : ``'welch'`` | ``'multitaper'`` | ``'auto'``
            Spectral estimation method. ``'welch'`` uses Welch's
            method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
            tapers :footcite:p:`Slepian1978`. ``'auto'`` (default) uses Welch's
            method for continuous data and multitaper for
            :class:`~mne.Epochs` or :class:`~mne.Evoked` data.
        dB : bool
            Whether to plot on a decibel scale. If ``True``, plots
            10 × log₁₀(spectral_power/Hz).
        layout : instance of Layout | None
            Layout instance specifying sensor positions (does not need to be
            specified for Neuromag data). If ``None`` (default), the layout is
            inferred from the data (if possible).
        color : str | tuple
            A matplotlib-compatible color to use for the curves. Defaults to
            white.
        fig_facecolor : str | tuple
            A matplotlib-compatible color to use for the figure background. Defaults to
            black.
        axis_facecolor : str | tuple
            A matplotlib-compatible color to use for the axis background.
            Defaults to black.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            be length 1 (for efficiency, subplots for each channel are simulated
            within a single :class:`~matplotlib.axes.Axes`
            object). Default is ``None``.
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
        **method_kw
            Additional keyword arguments passed to the spectral estimation
            function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
            for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
            for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
            and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
            that for Welch method if ``n_fft`` is unspecified its default will be
            the smaller of ``2048`` or the number of available time samples (taking into
            account ``tmin`` and ``tmax``), not ``256`` as in
            :func:`~mne.time_frequency.psd_array_welch`.
            Defaults to ``dict(n_fft=2048)``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure distributing one image per channel across sensor topography.
        """
        init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot_topo)
        return self.compute_psd(**init_kw).plot_topo(**plot_kw)

    @legacy(alt=".compute_psd().plot_topomap()")
    @verbose_static(
        "bands_psd_topo",
        "tmin_tmax_psd",
        "ch_type_topomap_psd",
        "proj_psd",
        "method_plot_psd_auto",
        "normalize_psd_topo",
        "agg_fun_psd_topo",
        "dB_plot_topomap",
        "sensors_topomap",
        "show_names_topomap",
        "mask_evoked_topomap",
        "mask_params_topomap",
        "mask_label_params_topomap",
        "contours_topomap",
        "outlines_topomap",
        "sphere_topomap_auto",
        "image_interp_topomap",
        "extrapolate_topomap",
        "border_topomap",
        "res_topomap",
        "size_topomap",
        "cmap_topomap",
        "vlim_plot_topomap_psd",
        "cnorm",
        "colorbar_topomap",
        "cbar_fmt_topomap_psd",
        "units_topomap",
        "axes_spectrum_plot_topomap",
        "show",
        "n_jobs",
        "method_kw_psd",
    )
    def plot_psd_topomap(
        self,
        bands=None,
        tmin=None,
        tmax=None,
        ch_type=None,
        *,
        proj=False,
        method="auto",
        normalize=False,
        agg_fun=None,
        dB=False,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        mask_label_params=None,
        contours=0,
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
        colorbar=True,
        cbar_fmt="auto",
        units=None,
        axes=None,
        show=True,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        """Plot scalp topography of PSD for chosen frequency bands.

        Parameters
        ----------
        bands : None | dict | list of tuple
            The frequencies or frequency ranges to plot. If a :class:`dict`, keys will
            be used as subplot titles and values should be either a single frequency
            (e.g., ``{'presentation rate': 6.5}``) or a length-two sequence of lower
            and upper frequency band edges (e.g., ``{'theta': (4, 8)}``). If a single
            frequency is provided, the plot will show the frequency bin that is closest
            to the requested value. If ``None`` (the default), expands to::

                bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
                         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
                         'Gamma (30-45 Hz)': (30, 45)}

            .. note::
               For backwards compatibility, :class:`tuples<tuple>` of length 2 or 3 are
               also accepted, where the last element of the tuple is the subplot title
               and the other entries are frequency values (a single value or band
               edges). New code should use :class:`dict` or ``None``.

            .. versionchanged:: 1.2
               Allow passing a dict and discourage passing tuples.
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the mean for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        method : ``'welch'`` | ``'multitaper'`` | ``'auto'``
            Spectral estimation method. ``'welch'`` uses Welch's
            method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
            tapers :footcite:p:`Slepian1978`. ``'auto'`` (default) uses Welch's
            method for continuous data and multitaper for
            :class:`~mne.Epochs` or :class:`~mne.Evoked` data.
        normalize : bool
            If True, each band will be divided by the total power. Defaults to
            False.
        agg_fun : callable
            The function used to aggregate over frequencies. Defaults to
            :func:`numpy.sum` if ``normalize=True``, else :func:`numpy.mean`.
        dB : bool
            Whether to plot on a decibel scale. If ``True``, plots
            10 × log₁₀(spectral_power/Hz), following the application of
            ``agg_fun``. Ignored if ``normalize=True``.
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
        mask : ndarray of bool, shape (n_channels, n_times) | None
            Array indicating channel-time combinations to highlight with a distinct
            plotting style (useful for, e.g. marking which channels at which times a
            statistical test of the data reaches significance).
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
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.
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
        vlim : tuple of length 2 | "joint"
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data. Elements of the :class:`tuple` may also be
            callable functions which take in a :class:`NumPy array <numpy.ndarray>` and
            return a scalar.

            If both entries are ``None``, the bounds are set at
            ± the maximum absolute value
            of the data (yielding a colormap with midpoint at 0), or
            ``(0, max(abs(data)))`` if the (possibly baselined) data are all-positive.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. If ``vlim="joint"``, will compute the colormap
            limits jointly across all topomaps of the same channel type (instead of
            separately for each topomap), using the min/max of the data for that
            channel type. Defaults to ``(None, None)``.
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.

            .. versionadded:: 1.2
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
            If ``'auto'``, is equivalent to '%0.3f' if ``dB=False`` and '%0.1f' if
            ``dB=True``. Defaults to ``'auto'``.
        units : str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the length of ``bands``. Default is ``None``.
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
        **method_kw
            Additional keyword arguments passed to the spectral estimation
            function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
            for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
            for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
            and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
            that for Welch method if ``n_fft`` is unspecified its default will be
            the smaller of ``2048`` or the number of available time samples (taking into
            account ``tmin`` and ``tmax``), not ``256`` as in
            :func:`~mne.time_frequency.psd_array_welch`.

        Returns
        -------
        fig : instance of Figure
            Figure showing one scalp topography per frequency band.
        """  # noqa: E501
        init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot_topomap)
        return self.compute_psd(**init_kw).plot_topomap(**plot_kw)

    def _set_legacy_nfft_default(self, tmin, tmax, method, method_kw):
        """Update method_kw with legacy n_fft default for plot_psd[_topo]().

        This method returns ``None`` and has a side effect of (maybe) updating
        the ``method_kw`` dict.
        """
        if method == "welch" and method_kw.get("n_fft") is None:
            tm = _time_mask(self.times, tmin, tmax, sfreq=self.info["sfreq"])
            method_kw["n_fft"] = min(np.sum(tm), 2048)


class BaseSpectrum(ContainsMixin, UpdateChannelsMixin):
    """Base class for Spectrum and EpochsSpectrum."""

    def __init__(
        self,
        inst,
        method,
        fmin,
        fmax,
        tmin,
        tmax,
        picks,
        exclude,
        proj,
        remove_dc,
        *,
        n_jobs,
        verbose=None,
        **method_kw,
    ):
        # arg checking
        self._sfreq = inst.info["sfreq"]
        if np.isfinite(fmax) and (fmax > self.sfreq / 2):
            raise ValueError(
                f"Requested fmax ({fmax} Hz) must not exceed ½ the sampling "
                f"frequency of the data ({0.5 * inst.info['sfreq']} Hz)."
            )
        # method
        self._inst_type = type(inst)
        method = _validate_method(method, _get_instance_type_string(self))
        psd_funcs = dict(welch=psd_array_welch, multitaper=psd_array_multitaper)
        # triage method and kwargs. partial() doesn't check validity of kwargs,
        # so we do it manually to save compute time if any are invalid.
        psd_funcs = dict(welch=psd_array_welch, multitaper=psd_array_multitaper)
        _check_method_kwargs(psd_funcs[method], method_kw, msg=f'PSD method "{method}"')
        self._psd_func = partial(psd_funcs[method], remove_dc=remove_dc, **method_kw)

        # apply proj if desired
        if proj:
            inst = inst.copy().apply_proj()
        self.inst = inst

        # prep times and picks
        self._time_mask = _time_mask(inst.times, tmin, tmax, sfreq=self.sfreq)
        self._picks = _picks_to_idx(
            inst.info, picks, "data", exclude, with_ref_meg=False
        )

        # add the info object. bads and non-data channels were dropped by
        # _picks_to_idx() so we update the info accordingly:
        self.info = pick_info(inst.info, sel=self._picks, copy=True)

        # assign some attributes
        self.preload = True  # needed for __getitem__, never False
        self._method = method
        # self._dims may also get updated by child classes
        self._dims = (
            "channel",
            "freq",
        )
        if method_kw.get("average", "") in (None, False):
            self._dims += ("segment",)
        if self._returns_complex_tapers(**method_kw):
            self._dims = self._dims[:-1] + ("taper",) + self._dims[-1:]
        # record data type (for repr and html_repr)
        self._data_type = (
            "Fourier Coefficients"
            if method_kw.get("output") == "complex"
            else "Power Spectrum"
        )
        # set nave (child constructor overrides this for Evoked input)
        self._nave = None

    def __eq__(self, other):
        """Test equivalence of two Spectrum instances."""
        return object_diff(vars(self), vars(other)) == ""

    def __getstate__(self):
        """Prepare object for serialization."""
        inst_type_str = _get_instance_type_string(self)
        out = dict(
            method=self.method,
            data=self._data,
            sfreq=self.sfreq,
            dims=self._dims,
            freqs=self.freqs,
            inst_type_str=inst_type_str,
            data_type=self._data_type,
            info=self.info,
            nave=self.nave,
            weights=self.weights,
        )
        return out

    def __setstate__(self, state):
        """Unpack from serialized format."""
        from ..epochs import Epochs
        from ..evoked import Evoked
        from ..io import Raw

        self._method = state["method"]
        self._data = state["data"]
        self._freqs = state["freqs"]
        self._dims = state["dims"]
        self._sfreq = state["sfreq"]
        self.info = Info(**_convert_old_birthday_format(state["info"]))
        self._data_type = state["data_type"]
        self._nave = state.get("nave")  # objs saved before #11282 won't have `nave`
        self._weights = state.get("weights")  # objs saved before #12747 won't have
        self.preload = True
        # instance type
        inst_types = dict(Raw=Raw, Epochs=Epochs, Evoked=Evoked, Array=np.ndarray)
        self._inst_type = inst_types[state["inst_type_str"]]

    def __repr__(self):
        """Build string representation of the Spectrum object."""
        inst_type_str = _get_instance_type_string(self)
        # shape & dimension names
        dims = " × ".join(
            [f"{dim[0]} {dim[1]}s" for dim in zip(self.shape, self._dims)]
        )
        freq_range = f"{self.freqs[0]:0.1f}-{self.freqs[-1]:0.1f} Hz"
        return (
            f"<{self._data_type} (from {inst_type_str}, "
            f"{self.method} method) | {dims}, {freq_range}>"
        )

    @repr_html
    def _repr_html_(self, caption=None):
        """Build HTML representation of the Spectrum object."""
        inst_type_str = _get_instance_type_string(self)
        units = [f"{ch_type}: {unit}" for ch_type, unit in self.units().items()]
        t = _get_html_template("repr", "spectrum.html.jinja")
        t = t.render(
            inst=self, computed_from=inst_type_str, units=units, filenames=None
        )
        return t

    def _check_values(self):
        """Check PSD results for correct shape and bad values."""
        assert len(self._dims) == self._data.ndim, (self._dims, self._data.ndim)
        assert self._data.shape == self._shape
        # TODO: should this be more fine-grained (report "chan X in epoch Y")?
        ch_dim = self._dims.index("channel")
        dims = list(range(self._data.ndim))
        dims.pop(ch_dim)
        # take min() across all but the channel axis
        # (if the abs becomes memory intensive we could iterate over channels)
        use_data = self._data
        if use_data.dtype.kind == "c":
            use_data = np.abs(use_data)
        bad_value = use_data.min(axis=tuple(dims)) == 0
        bad_value &= ~np.isin(self.ch_names, self.info["bads"])
        if bad_value.any():
            chs = np.array(self.ch_names)[bad_value].tolist()
            s = _pl(bad_value.sum())
            warn(f"Zero value in spectrum for channel{s} {', '.join(chs)}", UserWarning)

    def _returns_complex_tapers(self, **method_kw):
        return self.method == "multitaper" and method_kw.get("output") == "complex"

    def _compute_spectra(self, data, fmin, fmax, n_jobs, method_kw, verbose):
        # make the spectra
        result = self._psd_func(
            data, self.sfreq, fmin=fmin, fmax=fmax, n_jobs=n_jobs, verbose=verbose
        )
        # assign ._data (handling unaggregated multitaper output)
        if self._returns_complex_tapers(**method_kw):
            fourier_coefs, freqs, weights = result
            self._data = fourier_coefs
            self._weights = weights
        else:
            psds, freqs = result
            self._data = psds
            self._weights = None
        # assign properties (._data already assigned above)
        self._freqs = freqs
        # this is *expected* shape, it gets asserted later in _check_values()
        # (and then deleted afterwards)
        self._shape = (len(self.ch_names), len(self.freqs))
        # append n_welch_segments (use "" as .get() default since None considered valid)
        if method_kw.get("average", "") in (None, False):
            n_welch_segments = _compute_n_welch_segments(data.shape[-1], method_kw)
            self._shape += (n_welch_segments,)
        # insert n_tapers
        if self._returns_complex_tapers(**method_kw):
            self._shape = self._shape[:-1] + (self._weights.size,) + self._shape[-1:]
        # we don't need these anymore, and they make save/load harder
        del self._picks
        del self._psd_func
        del self._time_mask

    @property
    def _detrend_picks(self):
        """Provide compatibility with __iter__."""
        return list()

    @property
    def ch_names(self):
        return self.info["ch_names"]

    @property
    def data(self):
        return self._data

    @property
    def freqs(self):
        return self._freqs

    @property
    def method(self):
        return self._method

    @property
    def nave(self):
        return self._nave

    @nave.setter
    def nave(self, nave):
        self._nave = nave

    @property
    def weights(self):
        return self._weights

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def shape(self):
        return self._data.shape

    def copy(self):
        """Return copy of the Spectrum instance.

        Returns
        -------
        spectrum : instance of Spectrum
            A copy of the object.
        """
        return deepcopy(self)

    @fill_doc_static(
        "picks_good_data_noref", "exclude_spectrum_get_data", "fmin_fmax_psd"
    )
    def get_data(
        self, picks=None, exclude="bads", fmin=0, fmax=np.inf, return_freqs=False
    ):
        """Get spectrum data in NumPy array format.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        exclude : list of str | 'bads'
            Channel names to exclude. If ``'bads'``, channels
            in ``spectrum.info['bads']`` are excluded; pass an empty list to
            include all channels (including "bad" channels, if any).
        fmin, fmax : float
            The lower- and upper-bound on frequencies of interest. Default is
            ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
        return_freqs : bool
            Whether to return the frequency bin values for the requested
            frequency range. Default is ``False``.

        Returns
        -------
        data : array
            The requested data in a NumPy array.
        freqs : array
            The frequency values for the requested range. Only returned if
            ``return_freqs`` is ``True``.
        """
        picks = _picks_to_idx(
            self.info, picks, "data_or_ica", exclude=exclude, with_ref_meg=False
        )
        fmin_idx = np.searchsorted(self.freqs, fmin)
        fmax_idx = np.searchsorted(self.freqs, fmax, side="right")
        freq_picks = np.arange(fmin_idx, fmax_idx)
        freq_axis = self._dims.index("freq")
        chan_axis = self._dims.index("channel")
        # normally there's a risk of np.take reducing array dimension if there
        # were only one channel or frequency selected, but `_picks_to_idx`
        # always returns an array of picks, and np.arange always returns an
        # array of freq bin indices, so we're safe; the result will always be
        # 2D.
        data = self._data.take(picks, chan_axis).take(freq_picks, freq_axis)
        if return_freqs:
            freqs = self._freqs[fmin_idx:fmax_idx]
            return (data, freqs)
        return data

    @fill_doc_static(
        "picks_all_data_noref",
        "dB_spectrum_plot",
        "xscale_plot_psd",
        "color_plot_psd",
        "spatial_colors_psd",
        "sphere_topomap_auto",
        "exclude_spectrum_plot",
        "axes_spectrum_plot_topomap",
        "show",
    )
    def plot(
        self,
        *,
        picks=None,
        average=False,
        dB=True,
        amplitude=False,
        xscale="linear",
        ci="sd",
        ci_alpha=0.3,
        color="black",
        alpha=None,
        spatial_colors=True,
        sphere=None,
        exclude=(),
        axes=None,
        show=True,
    ):
        """Plot power or amplitude spectra.

        Separate plots are drawn for each channel type. When the data have been
        processed with a bandpass, lowpass or highpass filter, dashed lines (╎)
        indicate the boundaries of the filter. The line noise frequency is also
        indicated with a dashed line (⋮). If ``average=False``, the plot will
        be interactive, and click-dragging on the spectrum will generate a
        scalp topography plot for the chosen frequency range in a new figure.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.

            .. versionchanged:: 1.5
                In version 1.5, the default behavior changed so that all
                :term:`data channels` (not just "good" data channels) are shown by
                default.
        average : bool
            Whether to average across channels before plotting. If ``True``, interactive
            plotting of scalp topography is disabled, and parameters ``ci`` and
            ``ci_alpha`` control the style of the confidence band around the mean.
            Default is ``False``.
        dB : bool
            Whether to plot on a decibel scale. If ``True``, plots
            10 × log₁₀(spectral_power/Hz), or 20 × log₁₀(spectral amplitude/√Hz) if
            ``amplitude=True``.
        amplitude : bool
            Whether to plot an amplitude spectrum (``True``) or power spectrum
            (``False``).

            .. versionchanged:: 1.8
                In version 1.8, the default changed to ``amplitude=False``.
        xscale : 'linear' | 'log'
            Scale of the frequency axis. Default is ``'linear'``.
        ci : float | 'sd' | 'range' | None
            Type of confidence band drawn around the mean when ``average=True``. If
            ``'sd'`` the band spans ±1 standard deviation across channels. If
            ``'range'`` the band spans the range across channels at each frequency. If a
            :class:`float`, it indicates the (bootstrapped) confidence interval to
            display, and must satisfy ``0 < ci <= 100``. If ``None``, no band is drawn.
            Default is ``sd``.
        ci_alpha : float
            Opacity of the confidence band. Must satisfy ``0 <= ci_alpha <= 1``. Default
            is 0.3.
        color : str | tuple
            A matplotlib-compatible color to use. Has no effect when
            spatial_colors=True.
        alpha : float | None
            Opacity of the spectrum line(s). If :class:`float`, must satisfy
            ``0 <= alpha <= 1``. If ``None``, opacity will be ``1`` when
            ``average=True`` and ``0.1`` when ``average=False``. Default is ``None``.
        spatial_colors : bool
            Whether to color spectrum lines by channel location. Ignored if
            ``average=True``.
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
        exclude : list of str | 'bads'
            Channel names to exclude from being drawn. If ``'bads'``, channels
            in ``spectrum.info['bads']`` are excluded; pass an empty list to
            include all channels (including "bad" channels, if any).

            .. versionchanged:: 1.5
                In version 1.5, the default behavior changed from ``exclude='bads'`` to
                ``exclude=()``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the length of ``bands``. Default is ``None``.
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
            Figure with spectra plotted in separate subplots for each channel type.
        """  # noqa: E501
        # Must nest this _mpl_figure import because of the BACKEND global
        # stuff
        from ..viz._mpl_figure import _line_figure, _split_picks_by_type
        from ..viz.utils import (
            _plot_psd,
            plt_show,
        )

        # arg checking
        ci = _check_ci(ci)
        _check_option("xscale", xscale, ("log", "linear"))
        sphere = _check_sphere(sphere, self.info)
        # defaults
        scalings = _handle_default("scalings", None)
        titles = _handle_default("titles", None)
        units = _handle_default("units", None)

        _validate_type(amplitude, bool, "amplitude")
        estimate = "amplitude" if amplitude else "power"

        logger.info(f"Plotting {estimate} spectral density ({dB=}).")

        # split picks by channel type
        picks = _picks_to_idx(
            self.info, picks, "data", exclude=exclude, with_ref_meg=False
        )
        (picks_list, units_list, scalings_list, titles_list) = _split_picks_by_type(
            self, picks, units, scalings, titles
        )
        # prepare data (e.g. aggregate across dims, convert complex to power)
        psd_list = [
            self._prepare_data_for_plot(
                self._data.take(_p, axis=self._dims.index("channel"))
            )
            for _p in picks_list
        ]
        # initialize figure
        fig, axes = _line_figure(self, axes, picks=picks)
        # don't add ylabels & titles if figure has unexpected number of axes
        make_label = len(axes) == len(fig.axes)
        # Plot Frequency [Hz] xlabel only on the last axis
        xlabels_list = [False] * (len(axes) - 1) + [True]
        # plot
        _plot_psd(
            self,
            fig,
            self.freqs,
            psd_list,
            picks_list,
            titles_list,
            units_list,
            scalings_list,
            axes,
            make_label,
            color,
            area_mode=ci,
            area_alpha=ci_alpha,
            dB=dB,
            estimate=estimate,
            average=average,
            spatial_colors=spatial_colors,
            xscale=xscale,
            line_alpha=alpha,
            sphere=sphere,
            xlabels_list=xlabels_list,
        )
        plt_show(show, fig)
        return fig

    @fill_doc_static(
        "dB_spectrum_plot_topo",
        "layout_spectrum_plot_topo",
        "color_spectrum_plot_topo",
        "fig_facecolor",
        "axis_facecolor",
        "axes_spectrum_plot_topo",
        "show",
    )
    def plot_topo(
        self,
        *,
        dB=True,
        layout=None,
        color="w",
        fig_facecolor="k",
        axis_facecolor="k",
        axes=None,
        show=True,
    ):
        """Plot power spectral density, separately for each channel.

        Parameters
        ----------
        dB : bool
            Whether to plot on a decibel scale. If ``True``, plots
            10 × log₁₀(spectral_power/Hz).
        layout : instance of Layout | None
            Layout instance specifying sensor positions (does not need to be
            specified for Neuromag data). If ``None`` (default), the layout is
            inferred from the data (if possible).
        color : str | tuple
            A matplotlib-compatible color to use for the curves. Defaults to
            white.
        fig_facecolor : str | tuple
            A matplotlib-compatible color to use for the figure background. Defaults to
            black.
        axis_facecolor : str | tuple
            A matplotlib-compatible color to use for the axis background.
            Defaults to black.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            be length 1 (for efficiency, subplots for each channel are simulated
            within a single :class:`~matplotlib.axes.Axes`
            object). Default is ``None``.
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
            Figure distributing one image per channel across sensor topography.
        """
        from ..viz.topo import _plot_timeseries, _plot_timeseries_unified, _plot_topo
        from ..viz.utils import (
            plt_show,
        )

        if layout is None:
            layout = find_layout(self.info)

        psds, freqs = self.get_data(return_freqs=True)
        # prepare data (e.g. aggregate across dims, convert complex to power)
        psds = self._prepare_data_for_plot(psds)
        if dB:
            psds = 10 * np.log10(psds)
            y_label = "dB"
        else:
            y_label = "Power"
        show_func = partial(
            _plot_timeseries_unified, data=[psds], color=color, times=[freqs]
        )
        click_func = partial(_plot_timeseries, data=[psds], color=color, times=[freqs])
        picks = _pick_data_channels(self.info)
        info = pick_info(self.info, picks)
        fig = _plot_topo(
            info,
            times=freqs,
            show_func=show_func,
            click_func=click_func,
            layout=layout,
            axis_facecolor=axis_facecolor,
            fig_facecolor=fig_facecolor,
            x_label="Frequency (Hz)",
            unified=True,
            y_label=y_label,
            axes=axes,
        )
        plt_show(show)
        return fig

    @fill_doc_static(
        "bands_psd_topo",
        "ch_type_topomap_psd",
        "normalize_psd_topo",
        "agg_fun_psd_topo",
        "dB_plot_topomap",
        "sensors_topomap",
        "show_names_topomap",
        "mask_evoked_topomap",
        "mask_params_topomap",
        "mask_label_params_topomap",
        "contours_topomap",
        "outlines_topomap",
        "sphere_topomap_auto",
        "image_interp_topomap",
        "extrapolate_topomap",
        "border_topomap",
        "res_topomap",
        "size_topomap",
        "cmap_topomap",
        "vlim_plot_topomap_psd",
        "cnorm",
        "colorbar_topomap",
        "cbar_fmt_topomap_psd",
        "units_topomap",
        "axes_spectrum_plot_topomap",
        "show",
    )
    def plot_topomap(
        self,
        bands=None,
        ch_type=None,
        *,
        normalize=False,
        agg_fun=None,
        dB=False,
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
        colorbar=True,
        cbar_fmt="auto",
        units=None,
        axes=None,
        show=True,
    ):
        """Plot scalp topography of PSD for chosen frequency bands.

        Parameters
        ----------
        bands : None | dict | list of tuple
            The frequencies or frequency ranges to plot. If a :class:`dict`, keys will
            be used as subplot titles and values should be either a single frequency
            (e.g., ``{'presentation rate': 6.5}``) or a length-two sequence of lower
            and upper frequency band edges (e.g., ``{'theta': (4, 8)}``). If a single
            frequency is provided, the plot will show the frequency bin that is closest
            to the requested value. If ``None`` (the default), expands to::

                bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
                         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
                         'Gamma (30-45 Hz)': (30, 45)}

            .. note::
               For backwards compatibility, :class:`tuples<tuple>` of length 2 or 3 are
               also accepted, where the last element of the tuple is the subplot title
               and the other entries are frequency values (a single value or band
               edges). New code should use :class:`dict` or ``None``.

            .. versionchanged:: 1.2
               Allow passing a dict and discourage passing tuples.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the mean for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        normalize : bool
            If True, each band will be divided by the total power. Defaults to
            False.
        agg_fun : callable
            The function used to aggregate over frequencies. Defaults to
            :func:`numpy.sum` if ``normalize=True``, else :func:`numpy.mean`.
        dB : bool
            Whether to plot on a decibel scale. If ``True``, plots
            10 × log₁₀(spectral_power/Hz), following the application of
            ``agg_fun``. Ignored if ``normalize=True``.
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
        mask : ndarray of bool, shape (n_channels, n_times) | None
            Array indicating channel-time combinations to highlight with a distinct
            plotting style (useful for, e.g. marking which channels at which times a
            statistical test of the data reaches significance).
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
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.
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
        vlim : tuple of length 2 | "joint"
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data. Elements of the :class:`tuple` may also be
            callable functions which take in a :class:`NumPy array <numpy.ndarray>` and
            return a scalar.

            If both entries are ``None``, the bounds are set at
            ± the maximum absolute value
            of the data (yielding a colormap with midpoint at 0), or
            ``(0, max(abs(data)))`` if the (possibly baselined) data are all-positive.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. If ``vlim="joint"``, will compute the colormap
            limits jointly across all topomaps of the same channel type (instead of
            separately for each topomap), using the min/max of the data for that
            channel type. Defaults to ``(None, None)``.
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
            If ``'auto'``, is equivalent to '%0.3f' if ``dB=False`` and '%0.1f' if
            ``dB=True``. Defaults to ``'auto'``.
        units : str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the length of ``bands``. Default is ``None``.
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
        fig : instance of Figure
            Figure showing one scalp topography per frequency band.
        """  # noqa: E501
        from ..viz.topomap import (
            _make_head_outlines,
            _prepare_topomap_plot,
            plot_psds_topomap,
        )
        from ..viz.utils import (
            _get_plot_ch_type,
            _prepare_sensor_names,
        )

        ch_type = _get_plot_ch_type(self, ch_type)
        if units is None:
            units = _handle_default("units", None)
        unit = units[ch_type] if hasattr(units, "keys") else units
        scalings = _handle_default("scalings", None)
        scaling = scalings[ch_type]

        (
            picks,
            pos,
            merge_channels,
            names,
            ch_type,
            sphere,
            clip_origin,
        ) = _prepare_topomap_plot(self, ch_type, sphere=sphere)
        outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

        psds, freqs = self.get_data(picks=picks, return_freqs=True)
        # prepare data (e.g. aggregate across dims, convert complex to power)
        psds = self._prepare_data_for_plot(psds)
        psds *= scaling**2

        if merge_channels:
            psds, names = _merge_ch_data(psds, ch_type, names, method="mean")

        names = _prepare_sensor_names(names, show_names)
        return plot_psds_topomap(
            psds=psds,
            freqs=freqs,
            pos=pos,
            bands=bands,
            ch_type=ch_type,
            normalize=normalize,
            agg_fun=agg_fun,
            dB=dB,
            sensors=sensors,
            names=names,
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
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            unit=unit,
            axes=axes,
            show=show,
        )

    def _prepare_data_for_plot(self, data):
        # handle unaggregated Welch
        if "segment" in self._dims:
            logger.info("Aggregating Welch estimates (median) before plotting...")
            data = np.nanmedian(data, axis=self._dims.index("segment"))
        # handle unaggregated multitaper (also handles complex -> power)
        elif "taper" in self._dims:
            logger.info("Aggregating multitaper estimates before plotting...")
            data = _psd_from_mt(data, self.weights)

        # handle complex data (should only be Welch remaining)
        if np.iscomplexobj(data):
            data = (data * data.conj()).real  # Scaling may be slightly off

        # handle epochs
        if "epoch" in self._dims:
            # XXX TODO FIXME decide how to properly aggregate across repeated
            # measures (epochs) and non-repeated but correlated measures
            # (channels) when calculating stddev or a CI. For across-channel
            # aggregation, doi:10.1007/s10162-012-0321-8 used hotellings T**2
            # with a correction factor that estimated data rank using monte
            # carlo simulations; seems like we could use our own data rank
            # estimation methods to similar effect. Their exact approach used
            # complex spectra though, here we've already converted to power;
            # not sure if that makes an important difference? Anyway that
            # aggregation would need to happen in the _plot_psd function
            # though, not here... for now we just average like we always did.

            # only log message if averaging will actually have an effect
            if data.shape[0] > 1:
                logger.info("Averaging across epochs before plotting...")
            # epoch axis should always be the first axis
            data = data.mean(axis=0)

        return data

    @verbose_static("overwrite")
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save spectrum data to disk (in HDF5 format).

        Parameters
        ----------
        fname : path-like
            Path of file to save to.
        overwrite : bool
            If True (default False), overwrite the destination file if it
            exists.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        See Also
        --------
        mne.time_frequency.read_spectrum
        """
        _, write_hdf5 = _import_h5io_funcs()
        check_fname(fname, "spectrum", (".h5", ".hdf5"))
        fname = _check_fname(fname, overwrite=overwrite, verbose=verbose)
        out = self.__getstate__()
        write_hdf5(fname, out, overwrite=overwrite, title="mnepython", slash="replace")

    @verbose_static("picks_all", "copy_df", "long_format_df_spe", "df_return")
    def to_data_frame(
        self, picks=None, index=None, copy=True, long_format=False, *, verbose=None
    ):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        an additional column "freq" is added, unless ``index='freq'``
        (in which case frequency values form the DataFrame's index).

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
        index : str | list of str | None
            Kind of index to use for the DataFrame. If ``None``, a sequential
            integer index (:class:`pandas.RangeIndex`) will be used. If a
            :class:`str`, a :class:`pandas.Index` will be used (see Notes). If
            a list of two or more string values, a :class:`pandas.MultiIndex`
            will be used. Defaults to ``None``.
        copy : bool
            If ``True``, data will be copied. Otherwise data may be modified in place.
            Defaults to ``True``.
        long_format : bool
            If True, the DataFrame is returned in long format where each row is one
            observation of the signal at a unique combination of
            frequency and channel.
            For convenience, a ``ch_type`` column is added to facilitate
            subsetting the resulting DataFrame. Defaults to ``False``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        df : instance of pandas.DataFrame
            A dataframe suitable for usage with other statistical/plotting/analysis
            packages.

        Notes
        -----
        Valid values for ``index`` depend on whether the Spectrum was created
        from continuous data (:class:`~mne.io.Raw`, :class:`~mne.Evoked`) or
        discontinuous data (:class:`~mne.Epochs`). For continuous data, only
        ``None`` or ``'freq'`` is supported. For discontinuous data, additional
        valid values are ``'epoch'`` and ``'condition'``, or a :class:`list`
        comprising some of the valid string values (e.g.,
        ``['freq', 'epoch']``).
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # triage for Epoch-derived or unaggregated spectra
        from_epo = _get_instance_type_string(self) == "Epochs"
        unagg_welch = "segment" in self._dims
        unagg_mt = "taper" in self._dims
        # arg checking
        valid_index_args = ["freq"]
        if from_epo:
            valid_index_args += ["epoch", "condition"]
        index = _check_pandas_index_arguments(index, valid_index_args)
        # get data
        picks = _picks_to_idx(self.info, picks, "all", exclude=())
        data = self.get_data(picks)
        if copy:
            data = data.copy()
        # reshape
        if unagg_mt:
            data = np.moveaxis(data, self._dims.index("freq"), -2)
        if from_epo:
            n_epochs, n_picks, n_freqs = data.shape[:3]
        else:
            n_epochs, n_picks, n_freqs = (1,) + data.shape[:2]
        n_segs = data.shape[-1] if unagg_mt or unagg_welch else 1
        data = np.moveaxis(data, self._dims.index("channel"), -1)
        # at this point, should be ([epoch], freq, [segment/taper], channel)
        data = data.reshape(n_epochs * n_freqs * n_segs, n_picks)
        # prepare extra columns / multiindex
        mindex = list()
        default_index = list()
        if from_epo:
            rev_event_id = {v: k for k, v in self.event_id.items()}
            _conds = [rev_event_id[k] for k in self.events[:, 2]]
            conditions = np.repeat(_conds, n_freqs * n_segs)
            epoch_nums = np.repeat(self.selection, n_freqs * n_segs)
            mindex.extend([("condition", conditions), ("epoch", epoch_nums)])
            default_index.extend(["condition", "epoch"])
        freqs = np.tile(np.repeat(self.freqs, n_segs), n_epochs)
        mindex.append(("freq", freqs))
        default_index.append("freq")
        if unagg_mt or unagg_welch:
            name = "taper" if unagg_mt else "segment"
            seg_nums = np.tile(np.arange(n_segs), n_epochs * n_freqs)
            mindex.append((name, seg_nums))
            default_index.append(name)
        # build DataFrame
        df = _build_data_frame(
            self, data, picks, long_format, mindex, index, default_index=default_index
        )
        return df

    def units(self, latex=False):
        """Get the spectrum units for each channel type.

        Parameters
        ----------
        latex : bool
            Whether to format the unit strings as LaTeX. Default is ``False``.

        Returns
        -------
        units : dict
            Mapping from channel type to a string representation of the units
            for that channel type.
        """
        from ..viz.utils import (
            _format_units_psd,
        )

        units = _handle_default("si_units", None)
        return {
            ch_type: _format_units_psd(units[ch_type], power=True, latex=latex)
            for ch_type in sorted(self.get_channel_types(unique=True))
        }


@fill_doc_static(
    "method_psd_auto",
    "fmin_fmax_psd",
    "tmin_tmax_psd",
    "picks_good_data_noref",
    "exclude_psd",
    "proj_psd",
    "remove_dc",
    "reject_by_annotation_psd",
    "n_jobs",
    "verbose",
    "method_kw_psd",
    "info_not_none",
)
class Spectrum(BaseSpectrum):
    """Data object for spectral representations of continuous data.

    .. warning:: The preferred means of creating Spectrum objects from
                 continuous or averaged data is via the instance methods
                 :meth:`mne.io.Raw.compute_psd` or
                 :meth:`mne.Evoked.compute_psd`. Direct class instantiation
                 is not supported.

    Parameters
    ----------
    inst : instance of Raw or Evoked
        The data from which to compute the frequency spectrum.
    method : ``'welch'`` | ``'multitaper'`` | ``'auto'``
        Spectral estimation method. ``'welch'`` uses Welch's
        method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
        tapers :footcite:p:`Slepian1978`.
        ``'auto'`` (default) uses Welch's method for continuous data
        and multitaper for :class:`~mne.Evoked` data.
    fmin, fmax : float
        The lower- and upper-bound on frequencies of interest. Default is
        ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
    tmin, tmax : float | None
        First and last times to include, in seconds. ``None`` uses the first or
        last time present in the data. Default is ``tmin=None, tmax=None`` (all
        times).
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick good data channels
        (excluding reference MEG channels). Note that channels in ``info['bads']``
        *will be included* if their names or indices are explicitly provided.
    exclude : list of str | 'bads'
        Channel names to exclude. If ``'bads'``, channels
        in ``info['bads']`` are excluded; pass an empty list to
        include all channels (including "bad" channels, if any).
    proj : bool
        Whether to apply SSP projection vectors before spectral estimation.
        Default is ``False``.
    remove_dc : bool
        If ``True``, the mean is subtracted from each segment before computing
        its spectrum.
    reject_by_annotation : bool
        Whether to omit bad spans of data before spectral estimation. If
        ``True``, spans with annotations whose description begins with
        ``bad`` will be omitted.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.
    **method_kw
        Additional keyword arguments passed to the spectral estimation
        function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
        for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
        for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
        and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
        that for Welch method if ``n_fft`` is unspecified its default will be
        the smaller of ``2048`` or the number of available time samples (taking into
        account ``tmin`` and ``tmax``), not ``256`` as in
        :func:`~mne.time_frequency.psd_array_welch`.

    Attributes
    ----------
    ch_names : list
        The channel names.
    freqs : array
        Frequencies at which the amplitude, power, or fourier coefficients
        have been computed.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    method : ``'welch'`` | ``'multitaper'``
        The method used to compute the spectrum.
    nave : int | None
        The number of trials averaged together when generating the spectrum. ``None``
        indicates no averaging is known to have occurred.
    weights : array | None
        The weights for each taper. Only present if spectra computed with
        ``method='multitaper'`` and ``output='complex'``.

        .. versionadded:: 1.8

    See Also
    --------
    EpochsSpectrum
    SpectrumArray
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        inst,
        method,
        fmin,
        fmax,
        tmin,
        tmax,
        picks,
        exclude,
        proj,
        remove_dc,
        reject_by_annotation,
        *,
        n_jobs,
        verbose=None,
        **method_kw,
    ):
        from ..io import BaseRaw

        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        # do the basic setup
        super().__init__(
            inst,
            method,
            fmin,
            fmax,
            tmin,
            tmax,
            picks,
            exclude,
            proj,
            remove_dc,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )
        # get just the data we want
        if isinstance(self.inst, BaseRaw):
            start, stop = np.where(self._time_mask)[0][[0, -1]]
            rba = "NaN" if reject_by_annotation else None
            data = self.inst.get_data(
                self._picks, start, stop + 1, reject_by_annotation=rba
            )
            if np.any(np.isnan(data)) and method == "multitaper":
                raise NotImplementedError(
                    'Cannot use method="multitaper" when reject_by_annotation=True. '
                    'Please use method="welch" instead.'
                )

        else:  # Evoked
            data = self.inst.data[self._picks][:, self._time_mask]
        # set nave
        self._nave = getattr(inst, "nave", None)
        # compute the spectra
        self._compute_spectra(data, fmin, fmax, n_jobs, method_kw, verbose)
        # check for correct shape and bad values
        self._check_values()
        del self._shape  # calculated from self._data henceforth
        # save memory
        del self.inst

    @fill_doc_static("getitem_spectrum_return")
    def __getitem__(self, item):
        """Get Spectrum data.

        Parameters
        ----------
        item : int | slice | array-like
            Indexing is similar to a :class:`NumPy array<numpy.ndarray>`; see
            Notes.

        Returns
        -------
        data : ndarray
            The selected spectral data. Shape will be
            ``(n_channels, n_freqs)`` for normal power spectra,
            ``(n_channels, n_freqs, n_segments)`` for unaggregated
            Welch estimates, or ``(n_channels, n_tapers, n_freqs)``
            for unaggregated multitaper estimates.

        Notes
        -----
        Integer-, list-, and slice-based indexing is possible:

        - ``spectrum[0]`` gives all frequency bins in the first channel
        - ``spectrum[:3]`` gives all frequency bins in the first 3 channels
        - ``spectrum[[0, 2], 5]`` gives the value in the sixth frequency bin of
          the first and third channels
        - ``spectrum[(4, 7)]`` is the same as ``spectrum[4, 7]``.

        .. note::

           Unlike :class:`~mne.io.Raw` objects (which returns a tuple of the
           requested data values and the corresponding times), accessing
           :class:`~mne.time_frequency.Spectrum` values via subscript does
           **not** return the corresponding frequency bin values. If you need
           them, use ``spectrum.freqs[freq_indices]`` or
           ``spectrum.get_data(..., return_freqs=True)``.
        """
        from ..io import BaseRaw

        self._parse_get_set_params = partial(BaseRaw._parse_get_set_params, self)
        return BaseRaw._getitem(self, item, return_times=False)


def _check_data_shape(data, info, freqs, dim_names, weights, is_epoched):
    if data.ndim != len(dim_names):
        raise ValueError(
            f"Expected data to have {len(dim_names)} dimensions, got {data.ndim}."
        )

    allowed_dims = ["epoch", "channel", "freq", "segment", "taper"]
    if not is_epoched:
        allowed_dims.remove("epoch")
    # TODO maybe we should be nice and allow plural versions of each dimname?
    for dim in dim_names:
        _check_option("dim_names", dim, allowed_dims)
    if "channel" not in dim_names or "freq" not in dim_names:
        raise ValueError("Both 'channel' and 'freq' must be present in `dim_names`.")

    if list(dim_names).index("channel") != int(is_epoched):
        raise ValueError(
            f"'channel' must be the {'second' if is_epoched else 'first'} dimension of "
            "the data."
        )
    want_n_chan = _pick_data_channels(info, exclude=()).size
    got_n_chan = data.shape[list(dim_names).index("channel")]
    if got_n_chan != want_n_chan:
        raise ValueError(
            f"The number of channels in `data` ({got_n_chan}) must match the number of "
            f"good + bad data channels in `info` ({want_n_chan})."
        )

    # given we limit max array size and ensure channel & freq dims present, only one of
    # taper or segment can be present
    if "taper" in dim_names:
        if dim_names[-2] != "taper":  # _psd_from_mt assumes this (called when plotting)
            raise ValueError(
                "'taper' must be the second to last dimension of the data."
            )
        # expect weights for each taper
        actual = None if weights is None else weights.size
        expected = data.shape[list(dim_names).index("taper")]
        if actual != expected:
            raise ValueError(
                f"Expected size of `weights` to be {expected} to match 'n_tapers' in "
                f"`data`, got {actual}."
            )
    elif "segment" in dim_names and dim_names[-1] != "segment":
        raise ValueError("'segment' must be the last dimension of the data.")

    # freq being in wrong position ruled out by above checks
    want_n_freq = freqs.size
    got_n_freq = data.shape[list(dim_names).index("freq")]
    if got_n_freq != want_n_freq:
        raise ValueError(
            f"The number of frequencies in `data` ({got_n_freq}) must match the number "
            f"of elements in `freqs` ({want_n_freq})."
        )


@fill_doc_static("info_not_none", "freqs_tfr_array", "verbose", "notes_spectrum_array")
class SpectrumArray(Spectrum):
    """Data object for precomputed spectral data (in NumPy array format).

    Parameters
    ----------
    data : ndarray, shape (n_channels, [n_tapers], n_freqs, [n_segments])
        The spectra for each channel.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    dim_names : tuple of str
        The name of the dimensions in the data, in the order they occur. Must contain
        ``'channel'`` and ``'freq'``;  if data are unaggregated estimates, also include
        either a ``'segment'`` (e.g., Welch-like algorithms) or ``'taper'`` (e.g.,
        multitaper algorithms) dimension. If including ``'taper'``, you should also pass
        a ``weights`` parameter.

        .. versionadded:: 1.8
    weights : ndarray | None
        Weights for the ``'taper'`` dimension, if present (see ``dim_names``).

        .. versionadded:: 1.8
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.create_info
    mne.EvokedArray
    mne.io.RawArray
    EpochsSpectrumArray

    Notes
    -----
    If the data passed in is real-valued, it is assumed to represent spectral *power*
    (not amplitude, phase, etc), and downstream methods (such as
    :meth:`~mne.time_frequency.SpectrumArray.plot`) assume power data. If you pass in
    real-valued data that is not power, axis labels will be incorrect.

    If the data passed in is complex-valued, it is assumed to represent Fourier
    coefficients. Downstream plotting methods will treat the data as such, attempting to
    convert this to power before visualisation. If you pass in complex-valued data that
    is not Fourier coefficients, axis labels will be incorrect.

        .. versionadded:: 1.6
    """

    @_verbose_control
    def __init__(
        self,
        data,
        info,
        freqs,
        dim_names=("channel", "freq"),
        weights=None,
        *,
        verbose=None,
    ):
        # (channel, [taper], freq, [segment])
        _check_option("data.ndim", data.ndim, (2, 3))  # only allow one extra dimension

        _check_data_shape(data, info, freqs, dim_names, weights, is_epoched=False)

        self.__setstate__(
            dict(
                method="unknown",
                data=data,
                sfreq=info["sfreq"],
                dims=dim_names,
                freqs=freqs,
                inst_type_str="Array",
                data_type=(
                    "Fourier Coefficients"
                    if np.iscomplexobj(data)
                    else "Power Spectrum"
                ),
                info=info,
                weights=weights,
            )
        )


@fill_doc_static(
    "method_psd",
    "fmin_fmax_psd",
    "tmin_tmax_psd",
    "picks_good_data_noref",
    "exclude_psd",
    "proj_psd",
    "remove_dc",
    "n_jobs",
    "verbose",
    "method_kw_psd",
    "info_not_none",
)
class EpochsSpectrum(BaseSpectrum, GetEpochsMixin):
    """Data object for spectral representations of epoched data.

    .. warning:: The preferred means of creating Spectrum objects from Epochs
                 is via the instance method :meth:`mne.Epochs.compute_psd`.
                 Direct class instantiation is not supported.

    Parameters
    ----------
    inst : instance of Epochs
        The data from which to compute the frequency spectrum.
    method : ``'welch'`` | ``'multitaper'``
        Spectral estimation method. ``'welch'`` uses Welch's
        method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
        tapers :footcite:p:`Slepian1978`.
    fmin, fmax : float
        The lower- and upper-bound on frequencies of interest. Default is
        ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
    tmin, tmax : float | None
        First and last times to include, in seconds. ``None`` uses the first or
        last time present in the data. Default is ``tmin=None, tmax=None`` (all
        times).
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick good data channels
        (excluding reference MEG channels). Note that channels in ``info['bads']``
        *will be included* if their names or indices are explicitly provided.
    exclude : list of str | 'bads'
        Channel names to exclude. If ``'bads'``, channels
        in ``info['bads']`` are excluded; pass an empty list to
        include all channels (including "bad" channels, if any).
    proj : bool
        Whether to apply SSP projection vectors before spectral estimation.
        Default is ``False``.
    remove_dc : bool
        If ``True``, the mean is subtracted from each segment before computing
        its spectrum.
    n_jobs : int | None
        The number of jobs to run in parallel. If ``-1``, it is set
        to the number of CPU cores. Requires the :mod:`joblib` package.
        ``None`` (default) is a marker for 'unset' that will be interpreted
        as ``n_jobs=1`` (sequential execution) unless the call is performed under
        a :class:`joblib:joblib.parallel_config` context manager that sets another
        value for ``n_jobs``.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.
    **method_kw
        Additional keyword arguments passed to the spectral estimation
        function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
        for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
        for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
        and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
        that for Welch method if ``n_fft`` is unspecified its default will be
        the smaller of ``2048`` or the number of available time samples (taking into
        account ``tmin`` and ``tmax``), not ``256`` as in
        :func:`~mne.time_frequency.psd_array_welch`.

    Attributes
    ----------
    ch_names : list
        The channel names.
    freqs : array
        Frequencies at which the amplitude, power, or fourier coefficients
        have been computed.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    method : ``'welch'`` | ``'multitaper'``
        The method used to compute the spectrum.
    weights : array | None
        The weights for each taper. Only present if spectra computed with
        ``method='multitaper'`` and ``output='complex'``.

        .. versionadded:: 1.8

    See Also
    --------
    EpochsSpectrumArray
    Spectrum
    mne.Epochs.compute_psd

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        inst,
        method,
        fmin,
        fmax,
        tmin,
        tmax,
        picks,
        exclude,
        proj,
        remove_dc,
        *,
        n_jobs,
        verbose=None,
        **method_kw,
    ):
        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        # do the basic setup
        super().__init__(
            inst,
            method,
            fmin,
            fmax,
            tmin,
            tmax,
            picks,
            exclude,
            proj,
            remove_dc,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )
        # get just the data we want
        data = self.inst._get_data(picks=self._picks, on_empty="raise")[
            :, :, self._time_mask
        ]
        # compute the spectra
        self._compute_spectra(data, fmin, fmax, n_jobs, method_kw, verbose)
        self._dims = ("epoch",) + self._dims
        self._shape = (len(self.inst),) + self._shape
        # check for correct shape and bad values
        self._check_values()
        del self._shape
        # we need these for to_data_frame()
        self.event_id = self.inst.event_id.copy()
        self.events = self.inst.events.copy()
        self.selection = self.inst.selection.copy()
        # we need these for __getitem__()
        self.drop_log = deepcopy(self.inst.drop_log)
        self._metadata = self.inst.metadata
        # save memory
        del self.inst

    @fill_doc_static("getitem_epochspectrum_return")
    def __getitem__(self, item):
        """Subselect epochs from an EpochsSpectrum.

        Parameters
        ----------
        item : int | slice | array-like | str
            Access options are the same as for :class:`~mne.Epochs` objects,
            see the docstring of :meth:`mne.Epochs.__getitem__` for
            explanation.

        Returns
        -------
        data : ndarray
            The selected spectral data. Shape will be
            ``(n_epochs, n_channels, n_freqs)`` for normal power spectra,
            ``(n_epochs, n_channels, n_freqs, n_segments)`` for unaggregated
            Welch estimates, or ``(n_epochs, n_channels, n_tapers, n_freqs)``
            for unaggregated multitaper estimates.
        """
        return super().__getitem__(item)

    def __getstate__(self):
        """Prepare object for serialization."""
        out = super().__getstate__()
        out.update(
            metadata=self._metadata,
            drop_log=self.drop_log,
            event_id=self.event_id,
            events=self.events,
            selection=self.selection,
        )
        return out

    def __setstate__(self, state):
        """Unpack from serialized format."""
        super().__setstate__(state)
        self._metadata = state["metadata"]
        self.drop_log = state["drop_log"]
        self.event_id = state["event_id"]
        self.events = state["events"]
        self.selection = state["selection"]

    def average(self, method="mean"):
        """Average the spectra across epochs.

        Parameters
        ----------
        method : 'mean' | 'median' | callable
            How to aggregate spectra across epochs. If callable, must take a
            :class:`NumPy array<numpy.ndarray>` of shape
            ``(n_epochs, n_channels, n_freqs)`` and return an array of shape
            ``(n_channels, n_freqs)``. Default is ``'mean'``.

        Returns
        -------
        spectrum : instance of Spectrum
            The aggregated spectrum object.
        """
        from ..viz.utils import (
            _make_combine_callable,
        )

        _validate_type(method, ("str", "callable"), "method")
        method = _make_combine_callable(
            method, axis=0, valid=("mean", "median"), keepdims=False
        )
        if not callable(method):
            raise ValueError(
                '"method" must be a valid string or callable, '
                f"got a {type(method).__name__} ({method})."
            )
        # averaging unaggregated spectral estimates are not supported
        if "segment" in self._dims:
            raise NotImplementedError(
                "Averaging individual Welch segments across epochs is not "
                "supported. Consider averaging the signals before computing "
                "the Welch spectrum estimates."
            )
        if "taper" in self._dims:
            raise NotImplementedError(
                "Averaging multitaper tapers across epochs is not supported. Consider "
                "averaging the signals before computing the complex spectrum."
            )
        # serialize the object and update data, dims, and data type
        state = super().__getstate__()
        state["nave"] = state["data"].shape[0]
        state["data"] = method(state["data"])
        state["dims"] = state["dims"][1:]
        state["data_type"] = f"Averaged {state['data_type']}"
        defaults = dict(
            method=None,
            fmin=None,
            fmax=None,
            tmin=None,
            tmax=None,
            picks=None,
            exclude=(),
            proj=None,
            remove_dc=None,
            reject_by_annotation=None,
            n_jobs=None,
            verbose=None,
        )
        return Spectrum(state, **defaults)


@fill_doc_static(
    "info_not_none",
    "freqs_tfr_array",
    "events_epochs",
    "event_id",
    "verbose",
    "notes_spectrum_array",
)
class EpochsSpectrumArray(EpochsSpectrum):
    """Data object for precomputed epoched spectral data (in NumPy array format).

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, [n_tapers], n_freqs, [n_segments])
        The spectra for each channel in each epoch.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    events : ndarray of int, shape (n_events, 3)
        The identity and timing of experimental events, around which the epochs were
        created. See :term:`events` for more information. Events that don't match
        the events of interest as specified by ``event_id`` will be marked as
        ``IGNORED`` in the drop log.
    event_id : int | list of int | dict | str | list of str | None
        The id of the :term:`events` to consider. If dict, the keys can later be used to
        access associated :term:`events`. Example: dict(auditory=1, visual=3). If int, a
        dict will be created with the id as string. If a list of int, all :term:`events`
        with the IDs specified in the list are used. If a str or list of str, ``events``
        must be ``None`` to use annotations and then the IDs must be the name(s) of the
        annotations to use. If None, all :term:`events` will be used and a dict is
        created with string integer names corresponding to the event id integers.
    dim_names : tuple of str
        The name of the dimensions in the data, in the order they occur. Must contain
        ``'channel'`` and ``'freq'``;  if data are unaggregated estimates, also include
        either a ``'segment'`` (e.g., Welch-like algorithms) or ``'taper'`` (e.g.,
        multitaper algorithms) dimension. If including ``'taper'``, you should also pass
        a ``weights`` parameter.

        .. versionadded:: 1.8
    weights : ndarray | None
        Weights for the ``'taper'`` dimension, if present (see ``dim_names``).

        .. versionadded:: 1.8
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    mne.create_info
    mne.EpochsArray
    SpectrumArray

    Notes
    -----
    If the data passed in is real-valued, it is assumed to represent spectral *power*
    (not amplitude, phase, etc), and downstream methods (such as
    :meth:`~mne.time_frequency.SpectrumArray.plot`) assume power data. If you pass in
    real-valued data that is not power, axis labels will be incorrect.

    If the data passed in is complex-valued, it is assumed to represent Fourier
    coefficients. Downstream plotting methods will treat the data as such, attempting to
    convert this to power before visualisation. If you pass in complex-valued data that
    is not Fourier coefficients, axis labels will be incorrect.

        .. versionadded:: 1.6
    """

    @_verbose_control
    def __init__(
        self,
        data,
        info,
        freqs,
        events=None,
        event_id=None,
        dim_names=("epoch", "channel", "freq"),
        weights=None,
        *,
        verbose=None,
    ):
        # (epoch, channel, [taper], freq, [segment])
        _check_option("data.ndim", data.ndim, (3, 4))  # only allow one extra dimension

        if list(dim_names).index("epoch") != 0:
            raise ValueError("'epoch' must be the first dimension of `data`.")
        if events is not None and data.shape[0] != events.shape[0]:
            raise ValueError(
                f"The first dimension of `data` ({data.shape[0]}) must match the first "
                f"dimension of `events` ({events.shape[0]})."
            )

        _check_data_shape(data, info, freqs, dim_names, weights, is_epoched=True)

        self.__setstate__(
            dict(
                method="unknown",
                data=data,
                sfreq=info["sfreq"],
                dims=dim_names,
                freqs=freqs,
                inst_type_str="Array",
                data_type=(
                    "Fourier Coefficients"
                    if np.iscomplexobj(data)
                    else "Power Spectrum"
                ),
                info=info,
                events=events,
                event_id=event_id,
                metadata=None,
                selection=np.arange(data.shape[0]),
                drop_log=tuple(tuple() for _ in range(data.shape[0])),
                weights=weights,
            )
        )


def combine_spectrum(all_spectrum, weights="nave"):
    """Merge spectral data by weighted addition.

    Create a new :class:`mne.time_frequency.Spectrum` instance, using a combination of
    the supplied instances as its data. By default, the mean (weighted by trials) is
    used. Subtraction can be performed by passing negative weights (e.g., ``[1, -1]``).
    Data must have the same channels and the same frequencies.

    Parameters
    ----------
    all_spectrum : list of Spectrum
        The Spectrum objects.
    weights : list of float | str
        The weights to apply to the data of each :class:`~mne.time_frequency.Spectrum`
        instance, or a string describing the weighting strategy to apply: 'nave'
        computes sum-to-one weights proportional to each object’s nave attribute;
        'equal' weights each :class:`~mne.time_frequency.Spectrum` by
        ``1 / len(all_spectrum)``.

    Returns
    -------
    spectrum : Spectrum
        The new spectral data.

    Notes
    -----
    .. versionadded:: 1.10.0
    """
    spectrum = all_spectrum[0].copy()
    if isinstance(weights, str):
        if weights not in ("nave", "equal"):
            raise ValueError('Weights must be a list of float, or "nave" or "equal"')
        if weights == "nave":
            for s_ in all_spectrum:
                if s_.nave is None:
                    raise ValueError(f"The 'nave' attribute is not specified for {s_}")
            weights = np.array([e.nave for e in all_spectrum], float)
            weights /= weights.sum()
        else:  # == 'equal'
            weights = [1.0 / len(all_spectrum)] * len(all_spectrum)
    weights = np.array(weights, float)
    if weights.ndim != 1 or weights.size != len(all_spectrum):
        raise ValueError("Weights must be the same size as all_spectrum")

    ch_names = spectrum.ch_names
    for s_ in all_spectrum[1:]:
        assert s_.ch_names == ch_names, (
            f"{spectrum} and {s_} do not contain the same channels"
        )
        assert np.max(np.abs(s_.freqs - spectrum.freqs)) < 1e-7, (
            f"{spectrum} and {s_} do not contain the same frequencies"
        )

    # use union of bad channels
    bads = list(
        set(spectrum.info["bads"]).union(*(s_.info["bads"] for s_ in all_spectrum[1:]))
    )
    spectrum.info["bads"] = bads

    # combine spectral data
    spectrum._data = sum(w * s_.data for w, s_ in zip(weights, all_spectrum))
    if spectrum.nave is not None:
        spectrum._nave = max(
            int(1.0 / sum(w**2 / s_.nave for w, s_ in zip(weights, all_spectrum))), 1
        )
    return spectrum


def read_spectrum(fname):
    """Load a :class:`mne.time_frequency.Spectrum` object from disk.

    Parameters
    ----------
    fname : path-like
        Path to a spectrum file in HDF5 format, which should end with ``.h5`` or
        ``.hdf5``.

    Returns
    -------
    spectrum : instance of Spectrum
        The loaded Spectrum object.

    See Also
    --------
    mne.time_frequency.Spectrum.save
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, "path-like", "fname")
    fname = _check_fname(fname=fname, overwrite="read", must_exist=False)
    # read it in
    hdf5_dict = read_hdf5(fname, title="mnepython", slash="replace")
    defaults = dict(
        method=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        exclude=(),
        proj=None,
        remove_dc=None,
        reject_by_annotation=None,
        n_jobs=None,
        verbose=None,
    )
    Klass = EpochsSpectrum if "epoch" in hdf5_dict["dims"] else Spectrum
    return Klass(hdf5_dict, **defaults)


def _check_ci(ci):
    ci = "sd" if ci == "std" else ci  # be forgiving
    if _is_numeric(ci):
        if not (0 < ci <= 100):
            raise ValueError(f"ci must satisfy 0 < ci <= 100, got {ci}")
        ci /= 100.0
    else:
        _check_option("ci", ci, [None, "sd", "range"])
    return ci


def _compute_n_welch_segments(n_times, method_kw):
    # get default values from psd_array_welch
    _defaults = dict()
    for param in ("n_fft", "n_per_seg", "n_overlap"):
        _defaults[param] = signature(psd_array_welch).parameters[param].default
    # override defaults with user-specified values
    for key, val in _defaults.items():
        _defaults.update({key: method_kw.get(key, val)})
    # sanity check values / replace `None`s with real numbers
    n_fft, n_per_seg, n_overlap = _check_nfft(n_times, **_defaults)
    # compute expected number of segments
    step = n_per_seg - n_overlap
    return (n_times - n_overlap) // step


def _validate_method(method, instance_type):
    """Convert 'auto' to a real method name, and validate."""
    if method == "auto":
        method = "welch" if instance_type.startswith("Raw") else "multitaper"
    _check_option("method", method, ("welch", "multitaper"))
    return method
