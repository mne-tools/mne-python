"""Functions to plot raw M/EEG data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import OrderedDict

import numpy as np

from .._fiff.pick import _picks_to_idx, pick_channels, pick_types
from ..defaults import _handle_default
from ..filter import create_filter
from ..utils import _check_option, _get_stim_channel, _validate_type, legacy, verbose
from ..utils.spectrum import _split_psd_kwargs
from .utils import (
    _check_cov,
    _compute_scalings,
    _get_channel_plotting_order,
    _handle_decim,
    _handle_precompute,
    _make_event_color_dict,
    _shorten_path_from_middle,
)

_RAW_CLIP_DEF = 1.5


@verbose
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
    time_format="float",
    precompute=None,
    use_opengl=None,
    picks=None,
    *,
    theme=None,
    overview_mode=None,
    splash=True,
    verbose=None,
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

    bad_color : color object
        Color to make bad channels.
    %(event_color)s
        Defaults to ``'cyan'``.
    %(scalings)s
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
        The title of the window. If None, and either the filename of the
        raw object or '<unknown>' will be displayed as title.
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
    %(group_by_browse)s
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
    %(show_scrollbars)s
    %(show_scalebars)s

        .. versionadded:: 0.20.0
    %(time_format)s
    %(precompute)s
    %(use_opengl)s
    %(picks_all)s
    %(theme_pg)s

        .. versionadded:: 1.0
    %(overview_mode)s

        .. versionadded:: 1.1
    %(splash)s

        .. versionadded:: 1.6
    %(verbose)s

    Returns
    -------
    %(browser)s

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

    %(notes_2d_backend)s
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

    # handle event colors
    event_color_dict = _make_event_color_dict(event_color, events, event_id)

    # handle first_samp
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    event_id_rev = {v: k for k, v in (event_id or {}).items()}

    # generate window title; allow instances without a filename (e.g., ICA)
    if title is None:
        title = "<unknown>"
        fnames = list(tuple(raw.filenames))  # get a list of a copy of the filenames
        if len(fnames):
            title = fnames.pop(0)
            extra = f" ... (+ {len(fnames)} more)" if len(fnames) else ""
            title = f"{title}{extra}"
            if len(title) > 60:
                title = _shorten_path_from_middle(title)
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
        # display
        butterfly=butterfly,
        clipping=clipping,
        scrollbars_visible=show_scrollbars,
        scalebars_visible=show_scalebars,
        window_title=title,
        bgcolor=bgcolor,
        # Qt-specific
        precompute=precompute,
        use_opengl=use_opengl,
        theme=theme,
        overview_mode=overview_mode,
        splash=splash,
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
    block=False,
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
    block : bool
        Whether to halt program execution until the figure is closed.
        May not work on all systems / platforms. Defaults to False.
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
