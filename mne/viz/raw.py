"""Functions to plot raw M/EEG data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import OrderedDict

import numpy as np

from .._fiff.pick import _picks_to_idx, pick_channels, pick_types
from ..defaults import _RAW_CLIP_DEF, _handle_default
from ..filter import create_filter
from ..utils import (
    _check_option,
    _get_stim_channel,
    _validate_type,
    legacy,
    sizeof_fmt,
    verbose,
    verbose_static,
)
from ..utils.spectrum import _split_psd_kwargs
from .utils import (
    _check_cov,
    _compute_scalings,
    _get_channel_plotting_order,
    _handle_decim,
    _handle_precompute,
    _make_event_color_dict,
    _normalize_annotation_colors,
    _shorten_path_from_middle,
)


@verbose_static(
    "event_color",
    "scalings",
    "group_by_browse",
    "show_scrollbars",
    "show_scalebars",
    "show_zero_line",
    "time_format",
    "precompute",
    "use_opengl",
    "picks_all",
    "theme_pg",
    "overview_mode",
    "splash",
    "figure_class",
    "browser",
    "notes_2d_backend",
)
def plot_raw(
    raw,
    events=None,
    duration=10.0,
    start=0.0,
    n_channels=20,
    bgcolor="w",
    color=None,
    bad_color="lightgray",
    event_color="cyan",
    *,
    annotation_colors=None,
    annotation_regex=".*",
    scalings=None,
    remove_dc=True,
    order=None,
    show_options=False,
    title=None,
    show=True,
    block=False,
    highpass=None,
    lowpass=None,
    filtorder=4,
    clipping=_RAW_CLIP_DEF,
    show_first_samp=False,
    proj=True,
    group_by="type",
    butterfly=False,
    decim="auto",
    noise_cov=None,
    event_id=None,
    show_scrollbars=True,
    show_scalebars=True,
    show_zero_line=False,
    time_format="float",
    precompute=None,
    use_opengl=None,
    picks=None,
    theme=None,
    overview_mode=None,
    splash=True,
    verbose=None,
    figure_class=None,
):
    """Plot raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to plot.
    events : array | None
        Events to show with vertical bars.
    duration : float
        Time window (s) to plot. The lesser of this value and the duration
        of the raw file will be used.
    start : float
        Initial time to show (can be changed dynamically once plotted). If
        show_first_samp is True, then it is taken relative to
        ``raw.first_samp``.
    n_channels : int
        Number of channels to plot at once. Defaults to 20. The lesser of
        ``n_channels`` and ``len(raw.ch_names)`` will be shown.
        Has no effect if ``order`` is 'position', 'selection' or 'butterfly'.
    bgcolor : color object
        Color of the background.
    color : dict | color object | None
        Color for the data traces. If None, defaults to::

            dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m',
                 emg='k', ref_meg='steelblue', misc='k', stim='k',
                 resp='k', chpi='k')

        If a dict, keys can be channel *types* (e.g., ``'eeg'``) and/or
        channel *names* (e.g., ``'SFG, Left'``); name-based entries
        take precedence over type-based ones.

    bad_color : color object
        Color to make bad channels.
    event_color : color object | dict | None
        Color(s) to use for :term:`events`. To show all :term:`events` in the same
        color, pass any matplotlib-compatible color. To color events differently,
        pass a `dict` that maps event names or integer event numbers to colors
        (must include entries for *all* events, or include a "fallback" entry with
        key ``-1``). If ``None``, colors are chosen from the current Matplotlib
        color cycle.
    annotation_colors : dict | None
        A dictionary mapping annotation description strings to colors. Use this to
        override the default color assigned to specific annotation types (e.g.,
        ``dict(bad_segment='orange')``). Colors can be any valid Matplotlib color
        specification. Keys that do not match any annotation description in the data
        will trigger a warning. If ``None`` (default), automatic colors are used.

        .. versionadded:: 1.12.1
    annotation_regex : str
        A regex pattern applied to each annotation's label.
        Matching labels remain visible, non-matching labels are hidden.

        .. versionadded:: 1.11
    scalings : 'auto' | dict | None
        Scaling factors for the traces. If a dictionary where any
        value is ``'auto'``, the scaling factor is set to match the 99.5th
        percentile of the respective data. If ``'auto'``, all scalings (for all
        channel types) are set to ``'auto'``. If any values are ``'auto'`` and the
        data is not preloaded, a subset up to 100 MB will be loaded. If ``None``,
        defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                 resp=1, chpi=1e-4, whitened=1e2)

        .. note::
            A particular scaling value ``s`` corresponds to half of the visualized
            signal range around zero (i.e. from ``0`` to ``+s`` or from ``0`` to
            ``-s``). For example, the default scaling of ``20e-6`` (20µV) for EEG
            signals means that the visualized range will be 40 µV (20 µV in the
            positive direction and 20 µV in the negative direction).
    remove_dc : bool
        If True remove DC component when plotting data.
    order : array of int | None
        Order in which to plot data. If the array is shorter than the number of
        channels, only the given channels are plotted. If None (default), all
        channels are plotted. If ``group_by`` is ``'position'`` or
        ``'selection'``, the ``order`` parameter is used only for selecting the
        channels to be plotted.
    show_options : bool
        If True, a dialog for options related to projection is shown.
    title : str | None
        The title of the window. If None, the filename of the raw object is
        used; for in-memory instances without a filename (e.g.,
        `~mne.io.RawArray`), the class name and approximate size are used.
    show : bool
        Show figure if True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for setting bad channels on the fly by clicking on a line.
        May not work on all systems / platforms.
        (Only Qt) If you run from a script, this needs to
        be ``True`` or a Qt-eventloop needs to be started somewhere
        else in the script (e.g. if you want to implement the browser
        inside another Qt-Application).
    highpass : float | None
        Highpass to apply when displaying data.
    lowpass : float | None
        Lowpass to apply when displaying data.
        If highpass > lowpass, a bandstop rather than bandpass filter
        will be applied.
    filtorder : int
        Filtering order. 0 will use FIR filtering with MNE defaults.
        Other values will construct an IIR filter of the given order
        and apply it with :func:`~scipy.signal.filtfilt` (making the effective
        order twice ``filtorder``). Filtering may produce some edge artifacts
        (at the left and right edges) of the signals during display.

        .. versionchanged:: 0.18
           Support for ``filtorder=0`` to use FIR filtering.
    clipping : str | float | None
        If None, channels are allowed to exceed their designated bounds in
        the plot. If "clamp", then values are clamped to the appropriate
        range for display, creating step-like artifacts. If "transparent",
        then excessive values are not shown, creating gaps in the traces.
        If float, clipping occurs for values beyond the ``clipping`` multiple
        of their dedicated range, so ``clipping=1.`` is an alias for
        ``clipping='transparent'``.

        .. versionchanged:: 0.21
           Support for float, and default changed from None to 1.5.
    show_first_samp : bool
        If True, show time axis relative to the ``raw.first_samp``.
    proj : bool
        Whether to apply projectors prior to plotting (default is ``True``).
        Individual projectors can be enabled/disabled interactively (see
        Notes). This argument only affects the plot; use ``raw.apply_proj()``
        to modify the data stored in the Raw object.
    group_by : str
        How to group channels. ``'type'`` groups by channel type,
        ``'original'`` plots in the order of ch_names, ``'selection'`` uses
        Elekta's channel groupings (only works for Neuromag data),
        ``'position'`` groups the channels by the positions of the sensors.
        ``'selection'`` and ``'position'`` modes allow custom selections by
        using a lasso selector on the topomap. In butterfly mode, ``'type'``
        and ``'original'`` group the channels by type, whereas ``'selection'``
        and ``'position'`` use regional grouping. ``'type'`` and ``'original'``
        modes are ignored when ``order`` is not ``None``. Defaults to ``'type'``.
    butterfly : bool
        Whether to start in butterfly mode. Defaults to False.
    decim : int | 'auto'
        Amount to decimate the data during display for speed purposes.
        You should only decimate if the data are sufficiently low-passed,
        otherwise aliasing can occur. The 'auto' mode (default) uses
        the decimation that results in a sampling rate least three times
        larger than ``min(info['lowpass'], lowpass)`` (e.g., a 40 Hz lowpass
        will result in at least a 120 Hz displayed sample rate).
    noise_cov : instance of Covariance | str | None
        Noise covariance used to whiten the data while plotting.
        Whitened data channels are scaled by ``scalings['whitened']``,
        and their channel names are shown in italic.
        Can be a string to load a covariance from disk.
        See also :meth:`mne.Evoked.plot_white` for additional inspection
        of noise covariance properties when whitening evoked data.
        For data processed with SSS, the effective dependence between
        magnetometers and gradiometers may introduce differences in scaling,
        consider using :meth:`mne.Evoked.plot_white`.

        .. versionadded:: 0.16.0
    event_id : dict | None
        Event IDs used to show at event markers (default None shows
        the event numbers).

        .. versionadded:: 0.16.0
    show_scrollbars : bool
        Whether to show scrollbars when the plot is initialized. Can be toggled
        after initialization by pressing :kbd:`z` ("zen mode") while the plot
        window is focused. Default is ``True``.

        .. versionadded:: 0.19.0
    show_scalebars : bool
        Whether to show scale bars when the plot is initialized. Can be toggled
        after initialization by pressing :kbd:`s` while the plot window is focused.
        Default is ``True``.
    show_zero_line : bool
        Whether to show the zero line for each channel trace when the plot is
        initialized. The line always marks the true zero of the channel, even
        if the currently-visible window's mean has been subtracted for display
        (see ``remove_dc``). Can be toggled after initialization by pressing
        :kbd:`0` while the plot window is focused. Default is ``False``.

        .. versionadded:: 1.13
    time_format : 'float' | 'clock'
        Style of time labels on the horizontal axis. If ``'float'``, labels will be
        number of seconds from the start of the recording. If ``'clock'``,
        labels will show "clock time" (hours/minutes/seconds) inferred from
        ``raw.info['meas_date']``. Default is ``'float'``.

        .. versionadded:: 0.24
    precompute : bool | str
        Whether to load all data (not just the visible portion) into RAM and
        apply preprocessing (e.g., projectors) to the full data array in a separate
        processor thread, instead of window-by-window during scrolling. The default
        None uses the ``MNE_BROWSER_PRECOMPUTE`` variable, which defaults to
        ``'auto'``. ``'auto'`` compares available RAM space to the expected size of
        the precomputed data, and precomputes only if enough RAM is available.
        This is only used with the Qt backend.

        .. versionadded:: 0.24
        .. versionchanged:: 1.0
           Support for the ``MNE_BROWSER_PRECOMPUTE`` config variable.
    use_opengl : bool | None
        Whether to use OpenGL when rendering the plot (requires ``pyopengl``).
        May increase performance, but effect is dependent on system CPU and
        graphics hardware. Only works if using the Qt backend. Default is
        None, which will use False unless the user configuration variable
        ``MNE_BROWSER_USE_OPENGL`` is set to ``'true'``,
        see :func:`mne.set_config`.

        .. versionadded:: 0.24
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick all channels. Bad channels
        are included by default. Note that channels in ``info['bads']`` *will be
        included* if their names or indices are explicitly provided.
    theme : str | path-like
        Can be "auto", "light", or "dark" or a path-like to a
        custom stylesheet. For Dark-Mode and automatic Dark-Mode-Detection,
        `qdarkstyle <https://github.com/ColinDuquesnoy/QDarkStyleSheet>`__ and
        `darkdetect <https://github.com/albertosottile/darkdetect>`__,
        respectively, are required.
        If None (default), the config option MNE_BROWSER_THEME will be used,
        defaulting to "auto" if it's not found.

        For the ``"matplotlib"`` backend, only ``"light"``, ``"dark"``, and
        ``"auto"`` are supported. For the ``"qt"`` backend, a path-like to a
        custom stylesheet is also accepted.
    overview_mode : str | None
        Can be "channels", "empty", or "hidden" to set the overview bar mode
        for the ``'qt'`` backend. If None (default), the config option
        ``MNE_BROWSER_OVERVIEW_MODE`` will be used, defaulting to "channels"
        if it's not found.
    splash : bool
        If True (default), a splash screen is shown during the application
        startup. Only applicable to the ``qt`` backend.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.
    figure_class : class
        The backend specific ``MNEBrowseFigure`` class to use. This is typically
        used to pass a subclass in order to customize the plot. This parameter
        requires cooperation from the backend, and is currently only supported by
        the ``matplotlib`` backend.

    Returns
    -------
    fig : matplotlib.figure.Figure | mne_qt_browser.figure.MNEQtBrowser
        Browser instance.

    Notes
    -----
    The arrow keys (up/down/left/right) can typically be used to navigate
    between channels and time ranges, but this depends on the backend
    matplotlib is configured to use (e.g., mpl.use('TkAgg') should work). The
    left/right arrows will scroll by 25%% of ``duration``, whereas
    shift+left/shift+right will scroll by 100%% of ``duration``. The scaling
    can be adjusted with - and + (or =) keys. The viewport dimensions can be
    adjusted with page up/page down and home/end keys. Full screen mode can be
    toggled with the F11 key, and scrollbars can be hidden/shown by pressing
    'z'. Right-click a channel label to view its location. To mark or un-mark a
    channel as bad, click on a channel label or a channel trace. The changes
    will be reflected immediately in the raw object's ``raw.info['bads']``
    entry.

    If projectors are present, a button labelled "Prj" in the lower right
    corner of the plot window opens a secondary control window, which allows
    enabling/disabling specific projectors individually. This provides a means
    of interactively observing how each projector would affect the raw data if
    it were applied.

    Annotation mode is toggled by pressing 'a', butterfly mode by pressing
    'b', and whitening mode (when ``noise_cov is not None``) by pressing 'w'.
    By default, the channel means are removed when ``remove_dc`` is set to
    ``True``. This flag can be toggled by pressing 'd'.

    MNE-Python provides two different backends for browsing plots (i.e.,
    :meth:`raw.plot()<mne.io.Raw.plot>`, :meth:`epochs.plot()<mne.Epochs.plot>`,
    and :meth:`ica.plot_sources()<mne.preprocessing.ICA.plot_sources>`). One is
    based on :mod:`matplotlib`, and the other is based on
    :doc:`PyQtGraph<pyqtgraph:index>`. You can set the backend temporarily with the
    context manager :func:`mne.viz.use_browser_backend`, you can set it for the
    duration of a Python session using :func:`mne.viz.set_browser_backend`, and you
    can set the default for your computer via
    :func:`mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')<mne.set_config>`
    (or ``'qt'``).

    .. note:: For the PyQtGraph backend to run in IPython with ``block=False``
              you must run the magic command ``%gui qt5`` first.
    .. note:: To report issues with the PyQtGraph backend, please use the
              `issues <https://github.com/mne-tools/mne-qt-browser/issues>`_
              of ``mne-qt-browser``.
    """
    from ..annotations import _annotations_starts_stops
    from ..io import BaseRaw
    from ._figure import _get_browser

    info = raw.info.copy()
    sfreq = info["sfreq"]
    projs = info["projs"]
    # this will be an attr for which projectors are currently "on" in the plot
    projs_on = np.full_like(projs, proj, dtype=bool)
    # disable projs in info if user doesn't want to see them right away
    if not proj:
        with info._unlock():
            info["projs"] = list()

    # handle defaults / check arg validity
    color = _handle_default("color", color)
    scalings = _compute_scalings(scalings, raw, remove_dc=remove_dc, duration=duration)
    if scalings["whitened"] == "auto":
        scalings["whitened"] = 1.0
    _validate_type(raw, BaseRaw, "raw", "Raw")
    decim, picks_data = _handle_decim(info, decim, lowpass)
    noise_cov = _check_cov(noise_cov, info)
    units = _handle_default("units", None)
    unit_scalings = _handle_default("scalings", None)
    _check_option("group_by", group_by, ("selection", "position", "original", "type"))

    # clipping
    _validate_type(clipping, (None, "numeric", str), "clipping")
    if isinstance(clipping, str):
        _check_option(
            "clipping", clipping, ("clamp", "transparent"), extra="when a string"
        )
        clipping = 1.0 if clipping == "transparent" else clipping
    elif clipping is not None:
        clipping = float(clipping)

    # be forgiving if user asks for too much time
    duration = min(raw.times[-1], float(duration))

    # determine IIR filtering parameters
    if highpass is not None and highpass <= 0:
        raise ValueError(f"highpass must be > 0, got {highpass}")
    if highpass is None and lowpass is None:
        ba = filt_bounds = None
    else:
        filtorder = int(filtorder)
        if filtorder == 0:
            method = "fir"
            iir_params = None
        else:
            method = "iir"
            iir_params = dict(order=filtorder, output="sos", ftype="butter")
        ba = create_filter(
            np.zeros((1, int(round(duration * sfreq)))),
            sfreq,
            highpass,
            lowpass,
            method=method,
            iir_params=iir_params,
        )
        filt_bounds = _annotations_starts_stops(
            raw, ("edge", "bad_acq_skip"), invert=True
        )

    # compute event times in seconds
    if events is not None:
        event_times = (events[:, 0] - raw.first_samp).astype(float)
        event_times /= sfreq
        event_nums = events[:, 2]
    else:
        event_times = event_nums = None

    # determine trace order
    ch_names = np.array(raw.ch_names)
    ch_types = np.array(raw.get_channel_types())

    picks = _picks_to_idx(info, picks, none="all", exclude=())
    order = _get_channel_plotting_order(order, ch_types, picks=picks)
    n_channels = min(info["nchan"], n_channels, len(order))
    # adjust order based on channel selection, if needed
    selections = None
    if group_by in ("selection", "position"):
        selections = _setup_channel_selections(raw, group_by, order)
        order = np.concatenate(list(selections.values()))
        default_selection = list(selections)[0]
        n_channels = len(selections[default_selection])
    assert isinstance(order, np.ndarray)
    assert order.dtype.kind == "i"
    if order.size == 0:
        raise RuntimeError("No channels found to plot")

    # handle annotation_colors
    if annotation_colors is not None:
        annotation_colors = _normalize_annotation_colors(
            annotation_colors, raw.annotations
        )

    # handle event colors
    event_color_dict = _make_event_color_dict(event_color, events, event_id)

    # handle first_samp
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    event_id_rev = {v: k for k, v in (event_id or {}).items()}

    # generate window title; allow instances without a filename (e.g., RawArray)
    if title is None:
        # in-memory instances (e.g., RawArray) have filenames of (None,)
        fnames = [fname for fname in raw.filenames if fname is not None]
        if len(fnames):
            title = fnames.pop(0)
            extra = f" ... (+ {len(fnames)} more)" if len(fnames) else ""
            title = f"{title}{extra}"
            if len(title) > 60:
                title = _shorten_path_from_middle(title)
        else:  # give at least a hint about the data being shown
            title = f"{type(raw).__name__} (~{sizeof_fmt(raw._size)})"
    elif not isinstance(title, str):
        raise TypeError(f"title must be None or a string, got a {type(title)}")

    # gather parameters and initialize figure
    _validate_type(use_opengl, (bool, None), "use_opengl")
    precompute = _handle_precompute(precompute)
    params = dict(
        inst=raw,
        info=info,
        # channels and channel order
        ch_names=ch_names,
        ch_types=ch_types,
        ch_order=order,
        picks=order[:n_channels],
        n_channels=n_channels,
        picks_data=picks_data,
        group_by=group_by,
        ch_selections=selections,
        # time
        t_start=start,
        duration=duration,
        n_times=raw.n_times,
        first_time=first_time,
        time_format=time_format,
        decim=decim,
        # events
        event_color_dict=event_color_dict,
        event_times=event_times,
        event_nums=event_nums,
        event_id_rev=event_id_rev,
        annotation_regex=annotation_regex,
        # preprocessing
        projs=projs,
        projs_on=projs_on,
        apply_proj=proj,
        remove_dc=remove_dc,
        filter_coefs=ba,
        filter_bounds=filt_bounds,
        noise_cov=noise_cov,
        # scalings
        scalings=scalings,
        units=units,
        unit_scalings=unit_scalings,
        # colors
        ch_color_bad=bad_color,
        ch_color_dict=color,
        annotation_colors=annotation_colors,
        # display
        butterfly=butterfly,
        clipping=clipping,
        scrollbars_visible=show_scrollbars,
        scalebars_visible=show_scalebars,
        zero_line_visible=show_zero_line,
        window_title=title,
        bgcolor=bgcolor,
        # Qt-specific
        precompute=precompute,
        use_opengl=use_opengl,
        theme=theme,
        overview_mode=overview_mode,
        splash=splash,
        figure_class=figure_class,
    )

    fig = _get_browser(show=show, block=block, **params)

    return fig


@legacy(alt="Raw.compute_psd().plot()")
@verbose
def plot_raw_psd(
    raw,
    fmin=0,
    fmax=np.inf,
    tmin=None,
    tmax=None,
    proj=False,
    n_fft=None,
    n_overlap=0,
    reject_by_annotation=True,
    picks=None,
    ax=None,
    color="black",
    xscale="linear",
    area_mode="std",
    area_alpha=0.33,
    dB=True,
    estimate="power",
    show=True,
    n_jobs=None,
    average=False,
    line_alpha=None,
    spatial_colors=True,
    sphere=None,
    window="hamming",
    exclude="bads",
    verbose=None,
):
    """%(plot_psd_doc)s.

    Parameters
    ----------
    raw : instance of Raw
        The raw object.
    %(fmin_fmax_psd)s
    %(tmin_tmax_psd)s
    %(proj_psd)s
    n_fft : int | None
        Number of points to use in Welch FFT calculations. Default is ``None``,
        which uses the minimum of 2048 and the number of time points.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    %(reject_by_annotation_psd)s
    %(picks_good_data_noref)s
    %(ax_plot_psd)s
    %(color_plot_psd)s
    %(xscale_plot_psd)s
    %(area_mode_plot_psd)s
    %(area_alpha_plot_psd)s
    %(dB_plot_psd)s
    %(estimate_plot_psd)s
    %(show)s
    %(n_jobs)s
    %(average_plot_psd)s
    %(line_alpha_plot_psd)s
    %(spatial_colors_psd)s
    %(sphere_topomap_auto)s
    %(window_psd)s

        .. versionadded:: 0.22.0
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the bad channels
        are excluded. Pass an empty list to plot all channels (including
        channels marked "bad", if any).

        .. versionadded:: 0.24.0
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.

    Notes
    -----
    %(notes_plot_*_psd_func)s
    """
    from ..time_frequency import Spectrum

    init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot)
    return raw.compute_psd(**init_kw).plot(**plot_kw)


@legacy(alt="Raw.compute_psd().plot_topo()")
@verbose
def plot_raw_psd_topo(
    raw,
    tmin=0.0,
    tmax=None,
    fmin=0.0,
    fmax=100.0,
    proj=False,
    *,
    n_fft=2048,
    n_overlap=0,
    dB=True,
    layout=None,
    color="w",
    fig_facecolor="k",
    axis_facecolor="k",
    axes=None,
    block=None,
    show=True,
    n_jobs=None,
    verbose=None,
):
    """Plot power spectral density, separately for each channel.

    Parameters
    ----------
    raw : instance of io.Raw
        The raw instance to use.
    %(tmin_tmax_psd)s
    %(fmin_fmax_psd_topo)s
    %(proj_psd)s
    n_fft : int
        Number of points to use in Welch FFT calculations. Defaults to 2048.
    n_overlap : int
        The number of points of overlap between blocks. Defaults to 0
        (no overlap).
    %(dB_spectrum_plot_topo)s
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). If ``None`` (default), the layout is
        inferred from the data.
    color : str | tuple
        A matplotlib-compatible color to use for the curves. Defaults to white.
    fig_facecolor : str | tuple
        A matplotlib-compatible color to use for the figure background.
        Defaults to black.
    axis_facecolor : str | tuple
        A matplotlib-compatible color to use for the axis background.
        Defaults to black.
    %(axes_spectrum_plot_topo)s
    block : bool | None
        This parameter is deprecated and will be removed in MNE 1.15; blocking now
        follows Matplotlib's behavior (see ``show``).
    %(show)s
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure distributing one image per channel across sensor topography.
    """
    from ..time_frequency import Spectrum

    init_kw, plot_kw = _split_psd_kwargs(plot_fun=Spectrum.plot_topo)
    return raw.compute_psd(**init_kw).plot_topo(**plot_kw)


def _setup_channel_selections(raw, kind, order):
    """Get dictionary of channel groupings."""
    from ..channels import (
        _EEG_SELECTIONS,
        _SELECTIONS,
        _divide_to_regions,
        read_vectorview_selection,
    )

    _check_option("group_by", kind, ("position", "selection"))
    if kind == "position":
        selections_dict = _divide_to_regions(raw.info)
        keys = _SELECTIONS[1:]  # omit 'Vertex'
    else:  # kind == 'selection'
        from ..channels.channels import _get_ch_info

        (
            has_vv_mag,
            has_vv_grad,
            *_,
            has_neuromag_122_grad,
            has_csd_coils,
        ) = _get_ch_info(raw.info)
        if not (has_vv_grad or has_vv_mag or has_neuromag_122_grad):
            raise ValueError(
                "order='selection' only works for Neuromag "
                "data. Use order='position' instead."
            )
        selections_dict = OrderedDict()
        # get stim channel (if any)
        stim_ch = _get_stim_channel(None, raw.info, raise_error=False)
        stim_ch = stim_ch if len(stim_ch) else [""]
        stim_ch = pick_channels(raw.ch_names, stim_ch, ordered=False)
        # loop over regions
        keys = np.concatenate([_SELECTIONS, _EEG_SELECTIONS])
        for key in keys:
            channels = read_vectorview_selection(key, info=raw.info)
            picks = pick_channels(raw.ch_names, channels, ordered=False)
            picks = np.intersect1d(picks, order)
            if not len(picks):
                continue  # omit empty selections
            selections_dict[key] = np.concatenate([picks, stim_ch])
    # add misc channels
    misc = pick_types(
        raw.info,
        meg=False,
        eeg=False,
        stim=True,
        eog=True,
        ecg=True,
        emg=True,
        ref_meg=False,
        misc=True,
        resp=True,
        chpi=True,
        exci=True,
        ias=True,
        syst=True,
        seeg=False,
        bio=True,
        ecog=False,
        fnirs=False,
        dbs=False,
        temperature=True,
        gsr=True,
        exclude=(),
    )
    if len(misc) and np.isin(misc, order).any():
        selections_dict["Misc"] = misc
    return selections_dict
