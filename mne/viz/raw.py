"""Functions to plot raw M/EEG data."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

from functools import partial
from collections import OrderedDict

import numpy as np

from ..annotations import _annotations_starts_stops
from ..filter import create_filter
from ..io.pick import pick_types, _pick_data_channels, pick_info, pick_channels
from ..utils import verbose, _validate_type, _check_option
from ..time_frequency import psd_welch
from ..defaults import _handle_default
from .topo import _plot_topo, _plot_timeseries, _plot_timeseries_unified
from .utils import (plt_show, _compute_scalings, _handle_decim, _check_cov,
                    _shorten_path_from_middle,
                    _get_channel_plotting_order, _make_event_color_dict)

_RAW_CLIP_DEF = 1.5


@verbose
def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4,
             clipping=_RAW_CLIP_DEF,
             show_first_samp=False, proj=True, group_by='type',
             butterfly=False, decim='auto', noise_cov=None, event_id=None,
             show_scrollbars=True, show_scalebars=True, verbose=None):
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
    scalings : 'auto' | dict | None
        Scaling factors for the traces. If any fields in scalings are 'auto',
        the scaling factor is set to match the 99.5th percentile of a subset of
        the corresponding data. If scalings == 'auto', all scalings fields are
        set to 'auto'. If any fields are 'auto' and data is not preloaded, a
        subset of times up to 100mb will be loaded. If None, defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                 resp=1, chpi=1e-4, whitened=1e2)

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
    %(browse_group_by)s
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
    show_scalebars : bool
        Whether or not to show the scale bars. Defaults to True.

        .. versionadded:: 0.20.0
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Raw traces.

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
    """
    from ..io.base import BaseRaw
    from ._figure import _browse_figure

    info = raw.info.copy()
    sfreq = info['sfreq']
    projs = info['projs']
    # this will be an attr for which projectors are currently "on" in the plot
    projs_on = np.full_like(projs, proj, dtype=bool)
    # disable projs in info if user doesn't want to see them right away
    if not proj:
        info['projs'] = list()

    # handle defaults / check arg validity
    color = _handle_default('color', color)
    scalings = _compute_scalings(scalings, raw, remove_dc=remove_dc,
                                 duration=duration)
    if scalings['whitened'] == 'auto':
        scalings['whitened'] = 1.
    _validate_type(raw, BaseRaw, 'raw', 'Raw')
    decim, picks_data = _handle_decim(info, decim, lowpass)
    noise_cov = _check_cov(noise_cov, info)
    units = _handle_default('units', None)
    unit_scalings = _handle_default('scalings', None)
    _check_option('group_by', group_by,
                  ('selection', 'position', 'original', 'type'))

    # clipping
    _validate_type(clipping, (None, 'numeric', str), 'clipping')
    if isinstance(clipping, str):
        _check_option('clipping', clipping, ('clamp', 'transparent'),
                      extra='when a string')
        clipping = 1. if clipping == 'transparent' else clipping
    elif clipping is not None:
        clipping = float(clipping)

    # be forgiving if user asks for too much time
    duration = min(raw.times[-1], float(duration))

    # determine IIR filtering parameters
    if highpass is not None and highpass <= 0:
        raise ValueError(f'highpass must be > 0, got {highpass}')
    if highpass is None and lowpass is None:
        ba = filt_bounds = None
    else:
        filtorder = int(filtorder)
        if filtorder == 0:
            method = 'fir'
            iir_params = None
        else:
            method = 'iir'
            iir_params = dict(order=filtorder, output='sos', ftype='butter')
        ba = create_filter(np.zeros((1, int(round(duration * sfreq)))),
                           sfreq, highpass, lowpass, method=method,
                           iir_params=iir_params)
        filt_bounds = _annotations_starts_stops(
            raw, ('edge', 'bad_acq_skip'), invert=True)

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
    order = _get_channel_plotting_order(order, ch_types)
    n_channels = min(info['nchan'], n_channels, len(order))
    # adjust order based on channel selection, if needed
    selections = None
    if group_by in ('selection', 'position'):
        selections = _setup_channel_selections(raw, group_by, order)
        order = np.concatenate(list(selections.values()))
        default_selection = list(selections)[0]
        n_channels = len(selections[default_selection])

    # handle event colors
    event_color_dict = _make_event_color_dict(event_color, events, event_id)

    # handle first_samp
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    event_id_rev = {v: k for k, v in (event_id or {}).items()}

    # generate window title; allow instances without a filename (e.g., ICA)
    if title is None:
        title = '<unknown>'
        fnames = raw._filenames.copy()
        if len(fnames):
            title = fnames.pop(0)
            extra = f' ... (+ {len(fnames)} more)' if len(fnames) else ''
            title = f'{title}{extra}'
            if len(title) > 60:
                title = _shorten_path_from_middle(title)
    elif not isinstance(title, str):
        raise TypeError(f'title must be None or a string, got a {type(title)}')

    # gather parameters and initialize figure
    params = dict(inst=raw,
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
                  window_title=title)

    fig = _browse_figure(**params)
    fig._update_picks()

    # make channel selection dialog, if requested (doesn't work well in init)
    if group_by in ('selection', 'position'):
        fig._create_selection_fig()

    # update projector and data, and plot
    fig._update_projector()
    fig._update_trace_offsets()
    fig._update_data()
    fig._draw_traces()

    # plot annotations (if any)
    fig._setup_annotation_colors()
    fig._draw_annotations()

    # start with projectors dialog open, if requested
    if show_options:
        fig._toggle_proj_fig()

    # for blitting
    fig.canvas.flush_events()
    fig.mne.bg = fig.canvas.copy_from_bbox(fig.bbox)

    plt_show(show, block=block)
    return fig


@verbose
def plot_raw_psd(raw, fmin=0, fmax=np.inf, tmin=None, tmax=None, proj=False,
                 n_fft=None, n_overlap=0, reject_by_annotation=True,
                 picks=None, ax=None, color='black', xscale='linear',
                 area_mode='std', area_alpha=0.33, dB=True, estimate='auto',
                 show=True, n_jobs=1, average=False, line_alpha=None,
                 spatial_colors=True, sphere=None, window='hamming',
                 verbose=None):
    """%(plot_psd_doc)s.

    Parameters
    ----------
    raw : instance of Raw
        The raw object.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    tmin : float | None
        Start time to consider.
    tmax : float | None
        End time to consider.
    proj : bool
        Apply projection.
    n_fft : int | None
        Number of points to use in Welch FFT calculations.
        Default is None, which uses the minimum of 2048 and the
        number of time points.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    %(reject_by_annotation_raw)s
    %(plot_psd_picks_good_data)s
    ax : instance of Axes | None
        Axes to plot into. If None, axes will be created.
    %(plot_psd_color)s
    %(plot_psd_xscale)s
    %(plot_psd_area_mode)s
    %(plot_psd_area_alpha)s
    %(plot_psd_dB)s
    %(plot_psd_estimate)s
    %(show)s
    %(n_jobs)s
    %(plot_psd_average)s
    %(plot_psd_line_alpha)s
    %(plot_psd_spatial_colors)s
    %(topomap_sphere_auto)s
    %(window-psd)s

        .. versionadded:: 0.22.0
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.
    """
    from ._figure import _psd_figure
    # handle FFT
    if n_fft is None:
        if tmax is None or not np.isfinite(tmax):
            tmax = raw.times[-1]
        tmin = 0. if tmin is None else tmin
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
    # generate figure
    fig = _psd_figure(
        inst=raw, proj=proj, picks=picks, axes=ax, tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax, sphere=sphere, xscale=xscale, dB=dB,
        average=average, estimate=estimate, area_mode=area_mode,
        line_alpha=line_alpha, area_alpha=area_alpha, color=color,
        spatial_colors=spatial_colors, n_jobs=n_jobs, n_fft=n_fft,
        n_overlap=n_overlap, reject_by_annotation=reject_by_annotation,
        window=window)
    plt_show(show)
    return fig


@verbose
def plot_raw_psd_topo(raw, tmin=0., tmax=None, fmin=0., fmax=100., proj=False,
                      n_fft=2048, n_overlap=0, layout=None, color='w',
                      fig_facecolor='k', axis_facecolor='k', dB=True,
                      show=True, block=False, n_jobs=1, axes=None,
                      verbose=None):
    """Plot channel-wise frequency spectra as topography.

    Parameters
    ----------
    raw : instance of io.Raw
        The raw instance to use.
    tmin : float
        Start time for calculations. Defaults to zero.
    tmax : float | None
        End time for calculations. If None (default), the end of data is used.
    fmin : float
        Start frequency to consider. Defaults to zero.
    fmax : float
        End frequency to consider. Defaults to 100.
    proj : bool
        Apply projection. Defaults to False.
    n_fft : int
        Number of points to use in Welch FFT calculations. Defaults to 2048.
    n_overlap : int
        The number of points of overlap between blocks. Defaults to 0
        (no overlap).
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). If None (default), the correct layout is
        inferred from the data.
    color : str | tuple
        A matplotlib-compatible color to use for the curves. Defaults to white.
    fig_facecolor : str | tuple
        A matplotlib-compatible color to use for the figure background.
        Defaults to black.
    axis_facecolor : str | tuple
        A matplotlib-compatible color to use for the axis background.
        Defaults to black.
    dB : bool
        If True, transform data to decibels. Defaults to True.
    show : bool
        Show figure if True. Defaults to True.
    block : bool
        Whether to halt program execution until the figure is closed.
        May not work on all systems / platforms. Defaults to False.
    %(n_jobs)s
    axes : instance of matplotlib Axes | None
        Axes to plot into. If None, axes will be created.
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure distributing one image per channel across sensor topography.
    """
    if layout is None:
        from ..channels.layout import find_layout
        layout = find_layout(raw.info)

    psds, freqs = psd_welch(raw, tmin=tmin, tmax=tmax, fmin=fmin,
                            fmax=fmax, proj=proj, n_fft=n_fft,
                            n_overlap=n_overlap, n_jobs=n_jobs)
    if dB:
        psds = 10 * np.log10(psds)
        y_label = 'dB'
    else:
        y_label = 'Power'
    show_func = partial(_plot_timeseries_unified, data=[psds], color=color,
                        times=[freqs])
    click_func = partial(_plot_timeseries, data=[psds], color=color,
                         times=[freqs])
    picks = _pick_data_channels(raw.info)
    info = pick_info(raw.info, picks)

    fig = _plot_topo(info, times=freqs, show_func=show_func,
                     click_func=click_func, layout=layout,
                     axis_facecolor=axis_facecolor,
                     fig_facecolor=fig_facecolor, x_label='Frequency (Hz)',
                     unified=True, y_label=y_label, axes=axes)

    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)
    return fig


def _setup_channel_selections(raw, kind, order):
    """Get dictionary of channel groupings."""
    from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                             _divide_to_regions)
    from ..utils import _get_stim_channel
    _check_option('group_by', kind, ('position', 'selection'))
    if kind == 'position':
        selections_dict = _divide_to_regions(raw.info)
        keys = _SELECTIONS[1:]  # omit 'Vertex'
    else:  # kind == 'selection'
        from ..channels.channels import _get_ch_info
        (has_vv_mag, has_vv_grad, *_, has_neuromag_122_grad, has_csd_coils
         ) = _get_ch_info(raw.info)
        if not (has_vv_grad or has_vv_mag or has_neuromag_122_grad):
            raise ValueError("order='selection' only works for Neuromag "
                             "data. Use order='position' instead.")
        selections_dict = OrderedDict()
        # get stim channel (if any)
        stim_ch = _get_stim_channel(None, raw.info, raise_error=False)
        stim_ch = stim_ch if len(stim_ch) else ['']
        stim_ch = pick_channels(raw.ch_names, stim_ch)
        # loop over regions
        keys = np.concatenate([_SELECTIONS, _EEG_SELECTIONS])
        for key in keys:
            channels = read_selection(key, info=raw.info)
            picks = pick_channels(raw.ch_names, channels)
            picks = np.intersect1d(picks, order)
            if not len(picks):
                continue  # omit empty selections
            selections_dict[key] = np.concatenate([picks, stim_ch])
    # add misc channels
    misc = pick_types(raw.info, meg=False, eeg=False, stim=True, eog=True,
                      ecg=True, emg=True, ref_meg=False, misc=True,
                      resp=True, chpi=True, exci=True, ias=True, syst=True,
                      seeg=False, bio=True, ecog=False, fnirs=False,
                      exclude=())
    if len(misc) and np.in1d(misc, order).any():
        selections_dict['Misc'] = misc
    return selections_dict
