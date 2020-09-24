"""Functions to plot raw M/EEG data."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

import copy
from functools import partial
from collections import OrderedDict, defaultdict

import numpy as np

from ..annotations import _annotations_starts_stops
from ..filter import create_filter, _overlap_add_filter, _filtfilt
from ..io.pick import (pick_types, _pick_data_channels, pick_info,
                       pick_channels, _DATA_CH_TYPES_ORDER_DEFAULT)
from ..utils import verbose, _ensure_int, _validate_type, _check_option
from ..time_frequency import psd_welch
from ..defaults import _handle_default
from .topo import _plot_topo, _plot_timeseries, _plot_timeseries_unified
from .utils import (_prepare_mne_browse, figure_nobar, plt_show,
                    _get_figsize_from_config, _setup_browser_offsets,
                    _compute_scalings, plot_sensors, _handle_decim,
                    _setup_plot_projector, _check_cov, _set_ax_label_style,
                    _simplify_float, _check_psd_fmax, _set_window_title,
                    shorten_path_from_middle)


def _plot_update_raw_proj(params, bools):
    """Deal with changed proj."""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], params['whitened_ch_names'] = _setup_plot_projector(
        params['info'], params['noise_cov'], True, params['use_noise_cov'])
    params['update_fun']()
    params['plot_fun']()


def _update_raw_data(params):
    """Deal with time or proj changed."""
    start = params['t_start']
    start -= params['first_time']
    stop = params['raw'].time_as_index(start + params['duration'])[0]
    start = params['raw'].time_as_index(start)[0]
    data_picks = _pick_data_channels(params['raw'].info)
    data, times = params['raw'][:, start:stop]
    if params['projector'] is not None:
        data = np.dot(params['projector'], data)
    # remove DC
    if params['remove_dc'] is True:
        data -= np.mean(data, axis=1)[:, np.newaxis]
    if params['ba'] is not None:
        # filter with the same defaults as `raw.filter`
        starts, stops = params['filt_bounds']
        mask = (starts < stop) & (stops > start)
        starts = np.maximum(starts[mask], start) - start
        stops = np.minimum(stops[mask], stop) - start
        for start_, stop_ in zip(starts, stops):
            this_data = data[data_picks, start_:stop_]
            if isinstance(params['ba'], np.ndarray):  # FIR
                this_data = _overlap_add_filter(
                    this_data, params['ba'], copy=False)
            else:  # IIR
                this_data = _filtfilt(this_data, params['ba'], None, 1, False)
            data[data_picks, start_:stop_] = this_data
    # scale
    for di in range(data.shape[0]):
        ch_name = params['info']['ch_names'][di]
        # stim channels should be hard limited
        if params['types'][di] == 'stim':
            norm = float(max(data[di]))
        elif ch_name in params['whitened_ch_names'] and \
                ch_name not in params['info']['bads']:
            norm = params['scalings']['whitened']
        else:
            norm = params['scalings'][params['types'][di]]
        data[di] /= norm if norm != 0 else 1.
    params['data'] = data
    params['times'] = times


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
    event_color : color object | dict
        Color(s) to use for events. For all events in the same color, pass any
        matplotlib-compatible color. Can also be a `dict` mapping event numbers
        to colors, but if so it must include all events or include a "fallback"
        entry with key ``-1``.
    scalings : dict | None
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
    group_by : str
        How to group channels. ``'type'`` groups by channel type,
        ``'original'`` plots in the order of ch_names, ``'selection'`` uses
        Elekta's channel groupings (only works for Neuromag data),
        ``'position'`` groups the channels by the positions of the sensors.
        ``'selection'`` and ``'position'`` modes allow custom selections by
        using a lasso selector on the topomap. Pressing ``ctrl`` key while
        selecting allows appending to the current selection. Channels marked as
        bad appear with red edges on the topomap. In butterfly mode, ``'type'``
        and ``'original'`` group the channels by type, whereas ``'selection'``
        and ``'position'`` use regional grouping. ``'type'`` and ``'original'``
        modes are overridden with ``order`` keyword.
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
    if order is None:
        ch_type_order = _DATA_CH_TYPES_ORDER_DEFAULT
        order = [pick_idx for order_type in ch_type_order
                 for pick_idx, pick_type in enumerate(ch_types)
                 if order_type == pick_type]
    elif not isinstance(order, (np.ndarray, list, tuple)):
        raise ValueError('order should be array-like; got '
                         f'"{order}" ({type(order)}).')
    order = np.asarray(order)
    n_channels = min(info['nchan'], n_channels, len(order))
    # adjust order based on channel selection, if needed
    selections = None
    if group_by in ('selection', 'position'):
        selections = _setup_channel_selections(raw, group_by, order)
        order = np.concatenate(list(selections.values()))
        default_selection = list(selections)[0]
        n_channels = len(selections[default_selection])

    # if event_color is a dict
    if isinstance(event_color, dict):
        event_color = {_ensure_int(key, 'event_color key'): value
                       for key, value in event_color.items()}
        default = event_color.pop(-1, None)
        default_factory = None if default is None else lambda: default
        event_color_dict = defaultdict(default_factory)
        for key, value in event_color.items():
            if key < 1:
                raise KeyError('event_color keys must be strictly positive, '
                               f'or -1 (cannot use {key})')
            event_color_dict[key] = value
    # if event_color is a string or other MPL color-like thing
    else:
        event_color_dict = defaultdict(lambda: event_color)

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
                title = shorten_path_from_middle(title)
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
    fig._update_annotation_segments()
    fig._draw_annotations()

    # start with projectors dialog open, if requested
    if show_options:
        fig._toggle_proj_fig()

    # for blitting
    fig.canvas.flush_events()
    fig.mne.bg = fig.canvas.copy_from_bbox(fig.bbox)

    return fig


@verbose
def plot_raw_psd(raw, fmin=0, fmax=np.inf, tmin=None, tmax=None, proj=False,
                 n_fft=None, n_overlap=0, reject_by_annotation=True,
                 picks=None, ax=None, color='black', xscale='linear',
                 area_mode='std', area_alpha=0.33, dB=True, estimate='auto',
                 show=True, n_jobs=1, average=False, line_alpha=None,
                 spatial_colors=True, sphere=None, verbose=None):
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
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.
    """
    from .utils import _set_psd_plot_params, _plot_psd
    fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label, xlabels_list = _set_psd_plot_params(
            raw.info, proj, picks, ax, area_mode)
    _check_psd_fmax(raw, fmax)
    del ax
    psd_list = list()
    if n_fft is None:
        if tmax is None or not np.isfinite(tmax):
            tmax = raw.times[-1]
        tmin = 0. if tmin is None else tmin
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
    for picks in picks_list:
        psd, freqs = psd_welch(raw, tmin=tmin, tmax=tmax, picks=picks,
                               fmin=fmin, fmax=fmax, proj=proj, n_fft=n_fft,
                               n_overlap=n_overlap, n_jobs=n_jobs,
                               reject_by_annotation=reject_by_annotation)
        psd_list.append(psd)
    fig = _plot_psd(raw, fig, freqs, psd_list, picks_list, titles_list,
                    units_list, scalings_list, ax_list, make_label, color,
                    area_mode, area_alpha, dB, estimate, average,
                    spatial_colors, xscale, line_alpha, sphere, xlabels_list)
    plt_show(show)
    return fig


def _prepare_mne_browse_raw(params, title, bgcolor, color, bad_color, inds,
                            n_channels):
    """Set up the mne_browse_raw window."""
    import matplotlib as mpl

    figsize = _get_figsize_from_config()
    params['fig'] = figure_nobar(facecolor=bgcolor, figsize=figsize)
    _set_window_title(params['fig'], title or "Raw")
    # most of the axes setup is done in _prepare_mne_browse
    _prepare_mne_browse(params, xlabel='Time (s)')
    ax = params['ax']
    ax_hscroll = params['ax_hscroll']
    ax_vscroll = params['ax_vscroll']

    # populate vertical and horizontal scrollbars
    info = params['info']
    n_ch = len(inds)

    if 'fig_selection' in params:
        selections = params['selections']
        labels = [
            label._text for label in params['fig_selection'].radio.labels]
        # Flatten the selections dict to a list.
        sels = [selections[label] for label in labels]
        cis = [item for sublist in sels for item in sublist]

        for idx, ci in enumerate(cis):
            this_color = (bad_color if info['ch_names'][ci] in
                          info['bads'] else color)
            if isinstance(this_color, dict):
                this_color = this_color[params['types'][ci]]
            ax_vscroll.add_patch(mpl.patches.Rectangle((0, idx), 1, 1,
                                                       facecolor=this_color,
                                                       edgecolor=this_color))
        ax_vscroll.set_ylim(len(cis), 0)
        n_channels = max([len(selections[labels[0]]), n_channels])
    else:
        for ci in range(len(inds)):
            this_color = (bad_color if info['ch_names'][inds[ci]] in
                          info['bads'] else color)
            if isinstance(this_color, dict):
                this_color = this_color[params['types'][inds[ci]]]
            ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                       facecolor=this_color,
                                                       edgecolor=this_color))
        ax_vscroll.set_ylim(n_ch, 0)
    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       facecolor='w', edgecolor='w')
    ax_vscroll.add_patch(vsel_patch)
    params['vsel_patch'] = vsel_patch

    hsel_patch = mpl.patches.Rectangle((params['t_start'], 0),
                                       params['duration'], 1, edgecolor='k',
                                       facecolor=(0.75, 0.75, 0.75),
                                       alpha=0.25, linewidth=4, clip_on=False)
    ax_hscroll.add_patch(hsel_patch)
    params['hsel_patch'] = hsel_patch
    ax_hscroll.set_xlim(params['first_time'], params['first_time'] +
                        params['n_times'] / float(info['sfreq']))

    vertline_color = (0., 0.75, 0.)
    params['ax_vertline'] = ax.axvline(0, color=vertline_color, zorder=4)
    params['ax_vertline'].ch_name = ''
    params['vertline_t'] = ax_hscroll.text(params['first_time'], 1.2, '',
                                           color=vertline_color, fontsize=10,
                                           va='bottom', ha='right')
    params['ax_hscroll_vertline'] = ax_hscroll.axvline(0,
                                                       color=vertline_color,
                                                       zorder=2)
    # make shells for plotting traces
    _setup_browser_offsets(params, n_channels)
    ax.set_xlim(params['t_start'], params['t_start'] + params['duration'],
                False)

    params['lines'] = [ax.plot([np.nan], antialiased=True, linewidth=0.5)[0]
                       for _ in range(n_ch)]
    ax.set_yticklabels(
        ['X' * max([len(ch) for ch in info['ch_names']])] *
        len(params['offsets']))
    params['fig_annotation'] = None
    params['fig_help'] = None
    params['segment_line'] = None


def _plot_raw_traces(params, color, bad_color, event_lines=None,
                     event_color=None):
    """Plot raw traces."""
    from matplotlib.patches import Rectangle
    lines = params['lines']
    info = params['info']
    inds = params['inds']
    butterfly = params['butterfly']
    if butterfly:
        n_channels = len(params['offsets'])
        ch_start = 0
        offsets = params['offsets'][inds]
    else:
        n_channels = params['n_channels']
        ch_start = params['ch_start']
        offsets = params['offsets']
    params['bad_color'] = bad_color
    ax = params['ax']
    # Scalebars
    for bar in params.get('scalebars', {}).values():
        ax.lines.remove(bar)
    params['scalebars'] = dict()
    # delete event and annotation texts as well as scale bar texts
    params['ax'].texts = []
    # do the plotting
    tick_list = list()
    tick_colors = list()
    for ii in range(n_channels):
        ch_ind = ii + ch_start
        # let's be generous here and allow users to pass
        # n_channels per view >= the number of traces available
        if ii >= len(lines):
            break
        elif ch_ind < len(inds):
            # scale to fit
            ch_name = info['ch_names'][inds[ch_ind]]
            tick_list += [ch_name]
            offset = offsets[ii]
            this_type = params['types'][inds[ch_ind]]
            # do NOT operate in-place lest this get screwed up

            # apply user-supplied scale factor
            this_data = params['data'][inds[ch_ind]] * params['scale_factor']

            # set color
            this_color = bad_color if ch_name in info['bads'] else color
            if isinstance(this_color, dict):
                this_color = this_color[this_type]

            if inds[ch_ind] in params['data_picks']:
                this_decim = params['decim']
            else:
                this_decim = 1
            this_t = params['times'][::this_decim] + params['first_time']

            # clip to range (if relevant)
            if params['clipping'] == 'clamp':
                np.clip(this_data, -1, 1, out=this_data)
            elif params['clipping'] is not None:
                l, w = this_t[0], this_t[-1] - this_t[0]
                ylim = params['ax'].get_ylim()
                b = offset - params['clipping']  # max(, ylim[0])
                h = 2 * params['clipping']  # min(, ylim[1] - b)
                assert ylim[1] <= ylim[0]  # inverted
                b = max(b, ylim[1])
                h = min(h, ylim[0] - b)
                rect = Rectangle((l, b), w, h, transform=ax.transData)
                lines[ii].set_clip_path(rect)

            # subtraction here gets correct orientation for flipped ylim
            lines[ii].set_ydata(offset - this_data[..., ::this_decim])
            lines[ii].set_xdata(this_t)
            lines[ii].set_color(this_color)
            vars(lines[ii])['ch_name'] = ch_name
            vars(lines[ii])['def_color'] = color[this_type]
            this_z = 0 if ch_name in info['bads'] else 1
            if butterfly:
                if ch_name not in info['bads']:
                    if params['types'][ii] == 'mag':
                        this_z = 2
                    elif params['types'][ii] == 'grad':
                        this_z = 3
            else:
                # set label color
                this_color = (bad_color if ch_name in info['bads'] else
                              this_color)
                tick_colors.append(this_color)
            lines[ii].set_zorder(this_z)
            # add a scale bar
            if (params['show_scalebars'] and
                    this_type != 'stim' and
                    ch_name not in params['whitened_ch_names'] and
                    ch_name not in params['info']['bads'] and
                    this_type not in params['scalebars'] and
                    this_type in params['scalings'] and
                    this_type in params.get('unit_scalings', {}) and
                    this_type in params.get('units', {})):
                scale_color = '#AA3377'  # purple
                x = this_t[0]
                # This is what our data get multiplied by
                inv_norm = (
                    params['scalings'][this_type] *
                    params['unit_scalings'][this_type] *
                    2. /
                    params['scale_factor'])

                units = params['units'][this_type]
                bar = ax.plot([x, x], [offset - 1., offset + 1.],
                              color=scale_color, zorder=5, lw=4)[0]
                text = ax.text(x, offset + 1.,
                               '%s %s ' % (_simplify_float(inv_norm), units),
                               va='baseline', ha='right',
                               color=scale_color, zorder=5, size='xx-small')
                params['scalebars'][this_type] = bar
        else:
            # "remove" lines
            lines[ii].set_xdata([])
            lines[ii].set_ydata([])

    # deal with event lines
    if params['event_times'] is not None:
        # find events in the time window
        event_times = params['event_times']
        mask = np.logical_and(event_times >= params['times'][0],
                              event_times <= params['times'][-1])
        event_times = event_times[mask]
        event_nums = params['event_nums'][mask]
        # plot them with appropriate colors
        # go through the list backward so we end with -1, the catchall
        used = np.zeros(len(event_times), bool)
        ylim = params['ax'].get_ylim()
        for ev_num, line in zip(sorted(event_color.keys())[::-1],
                                event_lines[::-1]):
            mask = (event_nums == ev_num) if ev_num >= 0 else ~used
            assert not np.any(used[mask])
            used[mask] = True
            t = event_times[mask] + params['first_time']
            if len(t) > 0:
                xs = list()
                ys = list()
                for tt in t:
                    xs += [tt, tt, np.nan]
                    ys += [0, ylim[0], np.nan]
                line.set_xdata(xs)
                line.set_ydata(ys)
                line.set_zorder(0)
            else:
                line.set_xdata([])
                line.set_ydata([])

        # don't add event numbers for more than 50 visible events
        if len(event_times) <= 50:
            for ev_time, ev_num in zip(event_times, event_nums):
                if -1 in event_color or ev_num in event_color:
                    text = params['event_id_rev'].get(ev_num, ev_num)
                    params['ax'].text(ev_time, -0.1, text, fontsize=8,
                                      ha='center')

    if 'segments' in params:
        while len(params['ax'].collections) > 0:  # delete previous annotations
            params['ax'].collections.pop(-1)
        segments = params['segments']
        times = params['times']
        ylim = params['ax'].get_ylim()
        for idx, segment in enumerate(segments):
            if segment[0] > times[-1] + params['first_time']:
                break  # Since the segments are sorted by t_start
            if segment[1] < times[0] + params['first_time']:
                continue
            start = max(segment[0], times[0] + params['first_time'])
            end = min(times[-1] + params['first_time'], segment[1])
            dscr = params['raw'].annotations.description[idx]
            segment_color = params['segment_colors'][dscr]
            params['ax'].fill_betweenx(ylim, start, end, color=segment_color,
                                       alpha=0.3)
            params['ax'].text((start + end) / 2., ylim[1] - 0.1, dscr,
                              ha='center', color=segment_color)

    # finalize plot
    params['ax'].set_xlim(params['times'][0] + params['first_time'],
                          params['times'][0] + params['first_time'] +
                          params['duration'], False)
    if not butterfly:
        params['ax'].set_yticks(params['offsets'][:len(tick_list)])
        params['ax'].set_yticklabels(tick_list, rotation=0)
        _set_ax_label_style(params['ax'], params)
    else:
        tick_colors = ['k'] * len(params['ax'].get_yticks())
    for tick_color, tick in zip(tick_colors,
                                params['ax'].yaxis.get_ticklabels()):
        tick.set_color(tick_color)
    if 'fig_selection' not in params:
        params['vsel_patch'].set_y(params['ch_start'])
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


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


def _setup_browser_selection(raw, kind, selector=True):
    """Organize browser selections."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons
    from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                             _divide_to_regions)
    from ..utils import _get_stim_channel
    _check_option('group_by', kind, ('position, selection'))
    if kind == 'position':
        order = _divide_to_regions(raw.info)
        keys = _SELECTIONS[1:]  # no 'Vertex'
        kind = 'position'
    else:  # kind == 'selection'
        from ..io import RawFIF, RawArray
        if not isinstance(raw, (RawFIF, RawArray)):
            raise ValueError("order='selection' only works for Neuromag data. "
                             "Use order='position' instead.")
        order = dict()
        try:
            stim_ch = _get_stim_channel(None, raw.info)
        except ValueError:
            stim_ch = ['']
        keys = np.concatenate([_SELECTIONS, _EEG_SELECTIONS])
        stim_ch = pick_channels(raw.ch_names, stim_ch)
        for key in keys:
            channels = read_selection(key, info=raw.info)
            picks = pick_channels(raw.ch_names, channels)
            if len(picks) == 0:
                continue  # omit empty selections
            order[key] = np.concatenate([picks, stim_ch])

    misc = pick_types(raw.info, meg=False, eeg=False, stim=True, eog=True,
                      ecg=True, emg=True, ref_meg=False, misc=True, resp=True,
                      chpi=True, exci=True, ias=True, syst=True, seeg=False,
                      bio=True, ecog=False, fnirs=False, exclude=())
    if len(misc) > 0:
        order['Misc'] = misc
    keys = np.concatenate([keys, ['Misc']])
    if not selector:
        return order
    fig_selection = figure_nobar(figsize=(2, 6), dpi=80)
    _set_window_title(fig_selection, 'Selection')
    rax = plt.subplot2grid((6, 1), (2, 0), rowspan=4, colspan=1)
    topo_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
    keys = np.concatenate([keys, ['Custom']])
    order.update({'Custom': list()})  # custom selection with lasso
    plot_sensors(raw.info, kind='select', ch_type='all', axes=topo_ax,
                 ch_groups=kind, title='', show=False)
    fig_selection.radio = RadioButtons(rax, [key for key in keys
                                             if key in order.keys()])

    for circle in fig_selection.radio.circles:
        circle.set_radius(0.02)  # make them smaller to prevent overlap
        circle.set_edgecolor('gray')  # make sure the buttons are visible

    return order, fig_selection


def _setup_channel_selections(raw, kind, order):
    """Get dictionary of channel groupings."""
    from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                             _divide_to_regions)
    from ..utils import _get_stim_channel
    from ..io.pick import pick_channels, pick_types
    _check_option('group_by', kind, ('position', 'selection'))
    if kind == 'position':
        selections_dict = _divide_to_regions(raw.info)
        keys = _SELECTIONS[1:]  # omit 'Vertex'
    else:  # kind == 'selection'
        from ..io import RawFIF, RawArray
        if not isinstance(raw, (RawFIF, RawArray)):
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
