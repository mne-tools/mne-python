"""Functions to plot raw M/EEG data."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

import copy
from functools import partial

import numpy as np

from ..annotations import _annotations_starts_stops
from ..filter import create_filter, _overlap_add_filter, _filtfilt
from ..io.pick import (pick_types, _pick_data_channels, pick_info,
                       _PICK_TYPES_KEYS, pick_channels)
from ..utils import verbose, _ensure_int, _validate_type, _check_option
from ..time_frequency import psd_welch
from ..defaults import _handle_default
from .topo import _plot_topo, _plot_timeseries, _plot_timeseries_unified
from .utils import (_toggle_options, _toggle_proj, _prepare_mne_browse,
                    _plot_raw_onkey, figure_nobar, plt_show,
                    _plot_raw_onscroll, _mouse_click, _find_channel_idx,
                    _select_bads, _get_figsize_from_config,
                    _setup_browser_offsets, _compute_scalings, plot_sensors,
                    _radio_clicked, _set_radio_button, _handle_topomap_bads,
                    _change_channel_group, _plot_annotations, _setup_butterfly,
                    _handle_decim, _setup_plot_projector, _check_cov,
                    _set_ax_label_style, _draw_vert_line, _simplify_float,
                    _check_psd_fmax)


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


def _pick_bad_channels(event, params):
    """Select or drop bad channels onpick."""
    # Both bad lists are updated. params['info'] used for colors.
    if params['fig_annotation'] is not None:
        return
    bads = params['raw'].info['bads']
    params['info']['bads'] = _select_bads(event, params, bads)
    _plot_update_raw_proj(params, None)


@verbose
def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
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
        Color to use for events. Can also be a dict with
        ``{event_number: color}`` pairings. Use ``event_number==-1`` for
        any event numbers in the events list that are not in the dictionary.
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
    clipping : str | None
        If None, channels are allowed to exceed their designated bounds in
        the plot. If "clamp", then values are clamped to the appropriate
        range for display, creating step-like artifacts. If "transparent",
        then excessive values are not shown, creating gaps in the traces.
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
        using lasso selector on the topomap. Pressing ``ctrl`` key while
        selecting allows appending to the current selection. Channels marked as
        bad appear with red edges on the topomap. ``'type'`` and ``'original'``
        groups the channels by type in butterfly mode whereas ``'selection'``
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
    toggled with the F11 key. To mark or un-mark a channel as bad, click on a
    channel label or a channel trace. The changes will be reflected immediately
    in the raw object's ``raw.info['bads']`` entry.

    If projectors are present, a button labelled "Proj" in the lower right
    corner of the plot window opens a secondary control window, which allows
    enabling/disabling specific projectors individually. This provides a means
    of interactively observing how each projector would affect the raw data if
    it were applied.

    Annotation mode is toggled by pressing 'a', butterfly mode by pressing
    'b', and whitening mode (when ``noise_cov is not None``) by pressing 'w'.
    By default, the channel means are removed when ``remove_dc`` is set to
    ``True``. This flag can be toggled by pressing 'd'.
    """
    import matplotlib as mpl
    from ..io.base import BaseRaw
    color = _handle_default('color', color)
    scalings = _compute_scalings(scalings, raw, remove_dc=remove_dc,
                                 duration=duration)
    _validate_type(raw, BaseRaw, 'raw', 'Raw')
    n_channels = min(len(raw.info['chs']), n_channels)
    _check_option('clipping', clipping, [None, 'clamp', 'transparent'])
    duration = min(raw.times[-1], float(duration))

    # figure out the IIR filtering parameters
    sfreq = raw.info['sfreq']
    if highpass is not None and highpass <= 0:
        raise ValueError('highpass must be > 0, got %s' % (highpass,))
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

    # make a copy of info, remove projection (for now)
    info = raw.info.copy()
    projs = info['projs']
    info['projs'] = []
    n_times = raw.n_times

    # allow for raw objects without filename, e.g., ICA
    if title is None:
        title = raw._filenames
        if len(title) == 0:  # empty list or absent key
            title = '<unknown>'
        elif len(title) == 1:
            title = title[0]
        else:  # if len(title) > 1:
            title = '%s ... (+ %d more) ' % (title[0], len(title) - 1)
            if len(title) > 60:
                title = '...' + title[-60:]
    elif not isinstance(title, str):
        raise TypeError('title must be None or a string')
    if events is not None:
        event_times = events[:, 0].astype(float) - raw.first_samp
        event_times /= info['sfreq']
        event_nums = events[:, 2]
    else:
        event_times = event_nums = None

    # reorganize the data in plotting order
    # TODO Refactor this according to epochs.py
    inds = list()
    types = list()
    for t in ['grad', 'mag']:
        inds += [pick_types(info, meg=t, ref_meg=False, exclude=[])]
        types += [t] * len(inds[-1])
    for t in ['hbo', 'hbr', 'fnirs_raw', 'fnirs_od']:
        inds += [pick_types(info, meg=False, ref_meg=False, fnirs=t,
                            exclude=[])]
        types += [t] * len(inds[-1])
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    for key in _PICK_TYPES_KEYS:
        if key not in ['meg', 'fnirs']:
            pick_kwargs[key] = True
            inds += [pick_types(raw.info, **pick_kwargs)]
            types += [key] * len(inds[-1])
            pick_kwargs[key] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(info['ch_names']):
        raise RuntimeError('Some channels not classified, please report '
                           'this problem')

    # put them back to original or modified order for natural plotting
    reord = np.argsort(inds)
    types = [types[ri] for ri in reord]
    if isinstance(order, (np.ndarray, list, tuple)):
        # put back to original order first, then use new order
        inds = inds[reord][order]
    elif order is not None:
        raise ValueError('Unkown order, should be array-like. '
                         'Got "%s" (%s).' % (order, type(order)))

    if group_by in ['selection', 'position']:
        selections, fig_selection = _setup_browser_selection(raw, group_by)
        selections = {k: np.intersect1d(v, inds) for k, v in
                      selections.items()}
    elif group_by == 'original':
        if order is None:
            order = np.arange(len(inds))
            inds = inds[reord[:len(order)]]
    elif group_by != 'type':
        raise ValueError('Unknown group_by type %s' % group_by)

    if not isinstance(event_color, dict):
        event_color = {-1: event_color}
    event_color = {_ensure_int(key, 'event_color key'): event_color[key]
                   for key in event_color}
    for key in event_color:
        if key <= 0 and key != -1:
            raise KeyError('only key <= 0 allowed is -1 (cannot use %s)'
                           % key)
    decim, data_picks = _handle_decim(info, decim, lowpass)
    noise_cov = _check_cov(noise_cov, info)

    # set up projection and data parameters
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    event_id_rev = {val: key for key, val in (event_id or {}).items()}
    units = _handle_default('units', None)
    unit_scalings = _handle_default('scalings', None)

    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc, ba=ba,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, event_times=event_times, inds=inds,
                  event_nums=event_nums, clipping=clipping, fig_proj=None,
                  first_time=first_time, added_label=list(), butterfly=False,
                  group_by=group_by, orig_inds=inds.copy(), decim=decim,
                  data_picks=data_picks, event_id_rev=event_id_rev,
                  noise_cov=noise_cov, use_noise_cov=noise_cov is not None,
                  filt_bounds=filt_bounds, units=units, snap_annotations=False,
                  unit_scalings=unit_scalings, show_scrollbars=show_scrollbars,
                  show_scalebars=show_scalebars)

    if group_by in ['selection', 'position']:
        params['fig_selection'] = fig_selection
        params['selections'] = selections
        params['radio_clicked'] = partial(_radio_clicked, params=params)
        fig_selection.radio.on_clicked(params['radio_clicked'])
        lasso_callback = partial(_set_custom_selection, params=params)
        fig_selection.canvas.mpl_connect('lasso_event', lasso_callback)

    _prepare_mne_browse_raw(params, title, bgcolor, color, bad_color, inds,
                            n_channels)

    # plot event_line first so it's in the back
    event_lines = [params['ax'].plot([np.nan], color=event_color[ev_num])[0]
                   for ev_num in sorted(event_color.keys())]

    params['plot_fun'] = partial(_plot_raw_traces, params=params, color=color,
                                 bad_color=bad_color, event_lines=event_lines,
                                 event_color=event_color)

    _plot_annotations(raw, params)

    params['update_fun'] = partial(_update_raw_data, params=params)
    params['pick_bads_fun'] = partial(_pick_bad_channels, params=params)
    params['label_click_fun'] = partial(_label_clicked, params=params)
    params['scale_factor'] = 1.0
    # set up proj button
    opt_button = None
    if len(raw.info['projs']) > 0 and not raw.proj:
        ax_button = params['fig'].add_axes(params['proj_button_pos'])
        ax_button.set_axes_locator(params['proj_button_locator'])
        params['ax_button'] = ax_button
        params['apply_proj'] = proj
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
    # set up callbacks
    callback_key = partial(_plot_raw_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_pick)

    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_raw_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    params['callback_key'] = callback_key
    # have to store this, or it could get garbage-collected
    params['opt_button'] = opt_button
    params['update_vertline'] = partial(_draw_vert_line, params=params)

    # do initial plots
    callback_proj('none')

    # deal with projectors
    if show_options:
        _toggle_options(None, params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    # initialize the first selection set
    if group_by in ['selection', 'position']:
        _radio_clicked(fig_selection.radio.labels[0]._text, params)
        callback_selection_key = partial(_selection_key_press, params=params)
        callback_selection_scroll = partial(_selection_scroll, params=params)
        params['fig_selection'].canvas.mpl_connect('close_event',
                                                   callback_close)
        params['fig_selection'].canvas.mpl_connect('key_press_event',
                                                   callback_selection_key)
        params['fig_selection'].canvas.mpl_connect('scroll_event',
                                                   callback_selection_scroll)
    if butterfly:
        _setup_butterfly(params)

    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    # add MNE params dict to the resulting figure object so that parameters can
    # be modified after the figure has been created; this is useful e.g. to
    # remove the keyboard shortcut to close the figure with the 'Esc' key,
    # which can be done with
    #
    # fig._mne_params['close_key'] = None
    #
    # (assuming that the figure object is fig)
    params['fig']._mne_params = params

    return params['fig']


def _selection_scroll(event, params):
    """Handle scroll in selection dialog."""
    if event.step < 0:
        _change_channel_group(-1, params)
    elif event.step > 0:
        _change_channel_group(1, params)


def _selection_key_press(event, params):
    """Handle keys in selection dialog."""
    if event.key == 'down':
        _change_channel_group(-1, params)
    elif event.key == 'up':
        _change_channel_group(1, params)
    elif event.key == 'escape':
        _close_event(event, params)


def _close_event(event, params):
    """Handle closing of raw browser with selections."""
    import matplotlib.pyplot as plt
    if 'fig_selection' in params:
        plt.close(params['fig_selection'])
    for fig in ['fig_annotation', 'fig_help', 'fig_proj']:
        if params[fig] is not None:
            plt.close(params[fig])
    plt.close(params['fig'])


def _label_clicked(pos, params):
    """Select bad channels."""
    if params['butterfly']:
        return
    labels = params['ax'].yaxis.get_ticklabels()
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1])
    text = labels[line_idx].get_text()
    if len(text) == 0:
        return
    if 'fig_selection' in params:
        ch_idx = _find_channel_idx(text, params)
        _handle_topomap_bads(text, params)
    else:
        ch_idx = [params['ch_start'] + line_idx]
    bads = params['info']['bads']
    if text in bads:
        while text in bads:  # to make sure duplicates are removed
            bads.remove(text)
        color = vars(params['lines'][line_idx])['def_color']
        for idx in ch_idx:
            params['ax_vscroll'].patches[idx].set_color(color)
    else:
        bads.append(text)
        color = params['bad_color']
        for idx in ch_idx:
            params['ax_vscroll'].patches[idx].set_color(color)
    params['raw'].info['bads'] = bads
    _plot_update_raw_proj(params, None)


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
    reject_by_annotation : bool
        Whether to omit bad segments from the data while computing the
        PSD. If True, annotated segments with a description that starts
        with 'bad' are omitted. Has no effect if ``inst`` is an Epochs or
        Evoked object. Defaults to True.
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
    params['fig'].canvas.set_window_title(title or "Raw")
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
        labels = [l._text for l in params['fig_selection'].radio.labels]
        # Flatten the selections dict to a list.
        cis = [item for sublist in [selections[l] for l in labels] for item
               in sublist]

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

    ax_vscroll.set_title('Ch.')

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
    ax.set_yticklabels(['X' * max([len(ch) for ch in info['ch_names']])])
    params['fig_annotation'] = None
    params['fig_help'] = None
    params['segment_line'] = None


def _plot_raw_traces(params, color, bad_color, event_lines=None,
                     event_color=None):
    """Plot raw traces."""
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
    labels = ax.yaxis.get_ticklabels()
    # Scalebars
    for bar in params.get('scalebars', {}).values():
        ax.lines.remove(bar)
    params['scalebars'] = dict()
    # delete event and annotation texts as well as scale bar texts
    params['ax'].texts = []
    # do the plotting
    tick_list = list()
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

            # clip to range (if relevant)
            if params['clipping'] == 'transparent':
                this_data[np.abs(this_data) > 1] = np.nan
            elif params['clipping'] == 'clamp':
                np.clip(this_data, -1, 1, out=this_data)

            # set color
            this_color = bad_color if ch_name in info['bads'] else color
            if isinstance(this_color, dict):
                this_color = this_color[this_type]

            if inds[ch_ind] in params['data_picks']:
                this_decim = params['decim']
            else:
                this_decim = 1
            this_t = params['times'][::this_decim] + params['first_time']

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
                for label in labels:
                    label.set_color('black')
            else:
                # set label color
                this_color = (bad_color if ch_name in info['bads'] else
                              this_color)
                labels[ii].set_color(this_color)
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
        params['ax'].set_yticklabels(tick_list, rotation=0)
        _set_ax_label_style(params['ax'], params)
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


def _set_custom_selection(params):
    """Set custom selection by lasso selector."""
    chs = params['fig_selection'].lasso.selection
    if len(chs) == 0:
        return
    labels = [l._text for l in params['fig_selection'].radio.labels]
    inds = np.in1d(params['raw'].ch_names, chs)
    params['selections']['Custom'] = np.where(inds)[0]

    _set_radio_button(labels.index('Custom'), params=params)


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
    fig_selection.canvas.set_window_title('Selection')
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
