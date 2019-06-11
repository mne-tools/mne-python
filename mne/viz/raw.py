"""Functions to plot raw M/EEG data."""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

import copy
from functools import partial

import numpy as np

from ..annotations import _annotations_starts_stops
from ..filter import create_filter, _overlap_add_filter
from ..io.pick import (pick_types, _pick_data_channels, pick_info,
                       _PICK_TYPES_KEYS, pick_channels, channel_type,
                       _picks_to_idx)
from ..io.meas_info import create_info
from ..utils import (verbose, get_config, _ensure_int, _validate_type,
                     _check_option)
from ..time_frequency import psd_welch
from ..defaults import _handle_default
from .topo import _plot_topo, _plot_timeseries, _plot_timeseries_unified
from .utils import (_toggle_options, _toggle_proj, tight_layout,
                    _layout_figure, _plot_raw_onkey, figure_nobar, plt_show,
                    _plot_raw_onscroll, _mouse_click, _find_channel_idx,
                    _helper_raw_resize, _select_bads, _onclick_help,
                    _setup_browser_offsets, _compute_scalings, plot_sensors,
                    _radio_clicked, _set_radio_button, _handle_topomap_bads,
                    _change_channel_group, _plot_annotations, _setup_butterfly,
                    _handle_decim, _setup_plot_projector, _check_cov,
                    _set_ax_label_style, _draw_vert_line, warn)
from .evoked import _plot_lines


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
    from scipy.signal import filtfilt
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
            if isinstance(params['ba'], np.ndarray):
                data[data_picks, start_:stop_] = _overlap_add_filter(
                    data[data_picks, start_:stop_], params['ba'], copy=False)
            else:
                data[data_picks, start_:stop_] = filtfilt(
                    params['ba'][0], params['ba'][1],
                    data[data_picks, start_:stop_], axis=1, padlen=0)
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
    # clip
    if params['clipping'] == 'transparent':
        data[np.logical_or(data > 1, data < -1)] = np.nan
    elif params['clipping'] == 'clamp':
        data = np.clip(data, -1, 1, data)
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


def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
             show_first_samp=False, proj=True, group_by='type',
             butterfly=False, decim='auto', noise_cov=None, event_id=None):
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
        theh event numbers).

        .. versionadded:: 0.16.0

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Raw traces.

    Notes
    -----
    The arrow keys (up/down/left/right) can typically be used to navigate
    between channels and time ranges, but this depends on the backend
    matplotlib is configured to use (e.g., mpl.use('TkAgg') should work). The
    left/right arrows will scroll by 25% of ``duration``, whereas
    shift+left/shift+right will scroll by 100% of ``duration``. The scaling can
    be adjusted with - and + (or =) keys. The viewport dimensions can be
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
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.signal import butter
    from ..io.base import BaseRaw
    color = _handle_default('color', color)
    scalings = _compute_scalings(scalings, raw)
    _validate_type(raw, BaseRaw, 'raw', 'Raw')
    n_channels = min(len(raw.info['chs']), n_channels)
    _check_option('clipping', clipping, [None, 'clamp', 'transparent'])
    duration = min(raw.times[-1], float(duration))

    # figure out the IIR filtering parameters
    sfreq = raw.info['sfreq']
    nyq = sfreq / 2.
    if highpass is None and lowpass is None:
        ba = filt_bounds = None
    else:
        filtorder = int(filtorder)
        if highpass is not None and highpass <= 0:
            raise ValueError('highpass must be > 0, not %s' % highpass)
        if lowpass is not None and lowpass >= nyq:
            raise ValueError('lowpass must be < Nyquist (%s), not %s'
                             % (nyq, lowpass))
        if highpass is not None and lowpass is not None and \
                lowpass <= highpass:
            raise ValueError('lowpass (%s) must be > highpass (%s)'
                             % (lowpass, highpass))
        if filtorder == 0:
            ba = create_filter(np.zeros((1, int(round(duration * sfreq)))),
                               sfreq, highpass, lowpass)
        elif filtorder < 0:
            raise ValueError('filtorder (%s) must be >= 0' % filtorder)
        else:
            if highpass is None:
                Wn, btype = lowpass / nyq, 'lowpass'
            elif lowpass is None:
                Wn, btype = highpass / nyq, 'highpass'
            else:
                Wn, btype = [highpass / nyq, lowpass / nyq], 'bandpass'
            ba = butter(filtorder, Wn, btype, analog=False)
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
    for t in ['hbo', 'hbr']:
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
    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc, ba=ba,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, event_times=event_times, inds=inds,
                  event_nums=event_nums, clipping=clipping, fig_proj=None,
                  first_time=first_time, added_label=list(), butterfly=False,
                  group_by=group_by, orig_inds=inds.copy(), decim=decim,
                  data_picks=data_picks, event_id_rev=event_id_rev,
                  noise_cov=noise_cov, use_noise_cov=noise_cov is not None,
                  filt_bounds=filt_bounds)

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
    # set up callbacks
    opt_button = None
    if len(raw.info['projs']) > 0 and not raw.proj:
        ax_button = plt.subplot2grid((10, 10), (9, 9))
        params['ax_button'] = ax_button
        params['apply_proj'] = proj
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
    callback_key = partial(_plot_raw_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_pick)
    callback_resize = partial(_helper_raw_resize, params=params)
    params['fig'].canvas.mpl_connect('resize_event', callback_resize)

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
    _layout_figure(params)

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


_data_types = ('mag', 'grad', 'eeg', 'seeg', 'ecog')


def _set_psd_plot_params(info, proj, picks, ax, area_mode):
    """Set PSD plot params."""
    import matplotlib.pyplot as plt
    _check_option('area_mode', area_mode, [None, 'std', 'range'])
    picks = _picks_to_idx(info, picks)

    # XXX this could be refactored more with e.g., plot_evoked
    # XXX when it's refactored, Report._render_raw will need to be updated
    megs = ['mag', 'grad', False, False, False]
    eegs = [False, False, True, False, False]
    seegs = [False, False, False, True, False]
    ecogs = [False, False, False, False, True]
    titles = _handle_default('titles', None)
    units = _handle_default('units', None)
    scalings = _handle_default('scalings', None)
    picks_list = list()
    titles_list = list()
    units_list = list()
    scalings_list = list()
    for meg, eeg, seeg, ecog, name in zip(megs, eegs, seegs, ecogs,
                                          _data_types):
        these_picks = pick_types(info, meg=meg, eeg=eeg, seeg=seeg, ecog=ecog,
                                 ref_meg=False)
        these_picks = np.intersect1d(these_picks, picks)
        if len(these_picks) > 0:
            picks_list.append(these_picks)
            titles_list.append(titles[name])
            units_list.append(units[name])
            scalings_list.append(scalings[name])
    if len(picks_list) == 0:
        raise RuntimeError('No data channels found')
    if ax is not None:
        if isinstance(ax, plt.Axes):
            ax = [ax]
        if len(ax) != len(picks_list):
            raise ValueError('For this dataset with picks=None %s axes '
                             'must be supplied, got %s'
                             % (len(picks_list), len(ax)))
        ax_list = ax
    del picks

    fig = None
    if ax is None:
        fig = plt.figure()
        ax_list = list()
        for ii in range(len(picks_list)):
            # Make x-axes change together
            if ii > 0:
                ax_list.append(plt.subplot(len(picks_list), 1, ii + 1,
                                           sharex=ax_list[0]))
            else:
                ax_list.append(plt.subplot(len(picks_list), 1, ii + 1))
        make_label = True
    else:
        fig = ax_list[0].get_figure()
        make_label = len(ax_list) == len(fig.axes)

    return (fig, picks_list, titles_list, units_list, scalings_list,
            ax_list, make_label)


def _convert_psds(psds, dB, estimate, scaling, unit, ch_names):
    """Convert PSDs to dB (if necessary) and appropriate units.

    The following table summarizes the relationship between the value of
    parameters ``dB`` and ``estimate``, and the type of plot and corresponding
    units.

    | dB    | estimate    | plot | units             |
    |-------+-------------+------+-------------------|
    | True  | 'power'     | PSD  | amp**2/Hz (dB)    |
    | True  | 'amplitude' | ASD  | amp/sqrt(Hz) (dB) |
    | True  | 'auto'      | PSD  | amp**2/Hz (dB)    |
    | False | 'power'     | PSD  | amp**2/Hz         |
    | False | 'amplitude' | ASD  | amp/sqrt(Hz)      |
    | False | 'auto'      | ASD  | amp/sqrt(Hz)      |

    where amp are the units corresponding to the variable, as specified by
    ``unit``.
    """
    where = np.where(psds.min(1) <= 0)[0]
    dead_ch = ', '.join(ch_names[ii] for ii in where)
    if len(where) > 0:
        if dB:
            msg = "Infinite value in PSD for channel(s) %s. " \
                  "These channels might be dead." % dead_ch
        else:
            msg = "Zero value in PSD for channel(s) %s. " \
                  "These channels might be dead." % dead_ch
        warn(msg, UserWarning)

    if estimate == 'auto':
        estimate = 'power' if dB else 'amplitude'

    if estimate == 'amplitude':
        np.sqrt(psds, out=psds)
        psds *= scaling
        ylabel = r'$\mathrm{%s / \sqrt{Hz}}$' % unit
    else:
        psds *= scaling * scaling
        ylabel = r'$\mathrm{%s^2/Hz}$' % unit

    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
        psds *= 10
        ylabel += r'$\ \mathrm{(dB)}$'

    return ylabel


@verbose
def plot_raw_psd(raw, tmin=0., tmax=np.inf, fmin=0, fmax=np.inf, proj=False,
                 n_fft=None, picks=None, ax=None, color='black',
                 area_mode='std', area_alpha=0.33, n_overlap=0,
                 dB=True, estimate='auto', average=False, show=True, n_jobs=1,
                 line_alpha=None, spatial_colors=None, xscale='linear',
                 reject_by_annotation=True, verbose=None):
    """Plot the power spectral density across channels.

    Different channel types are drawn in sub-plots. When the data has been
    processed with a bandpass, lowpass or highpass filter, dashed lines
    indicate the boundaries of the filter (--). The line noise frequency is
    also indicated with a dashed line (-.).

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to use.
    tmin : float
        Start time for calculations.
    tmax : float
        End time for calculations.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    proj : bool
        Apply projection.
    n_fft : int | None
        Number of points to use in Welch FFT calculations.
        Default is None, which uses the minimum of 2048 and the
        number of time points.
    %(picks_good_data)s
        Cannot be None if `ax` is supplied. If both
        `picks` and `ax` are None, separate subplots will be created for
        each standard channel type (`mag`, `grad`, and `eeg`).
    ax : instance of matplotlib Axes | None
        Axes to plot into. If None, axes will be created.
    color : str | tuple
        A matplotlib-compatible color to use. Has no effect when
        spatial_colors=True.
    area_mode : str | None
        Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
        will be plotted. If 'range', the min and max (across channels) will be
        plotted. Bad channels will be excluded from these calculations.
        If None, no area will be plotted. If average=False, no area is plotted.
    area_alpha : float
        Alpha for the area.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    dB : bool
        Plot Power Spectral Density (PSD), in units (amplitude**2/Hz (dB)) if
        ``dB=True``, and ``estimate='power'`` or ``estimate='auto'``. Plot PSD
        in units (amplitude**2/Hz) if ``dB=False`` and,
        ``estimate='power'``. Plot Amplitude Spectral Density (ASD), in units
        (amplitude/sqrt(Hz)), if ``dB=False`` and ``estimate='amplitude'`` or
        ``estimate='auto'``. Plot ASD, in units (amplitude/sqrt(Hz) (db)), if
        ``dB=True`` and ``estimate='amplitude'``.
    estimate : str, {'auto', 'power', 'amplitude'}
        Can be "power" for power spectral density (PSD), "amplitude" for
        amplitude spectrum density (ASD), or "auto" (default), which uses
        "power" when dB is True and "amplitude" otherwise.
    average : bool
        If False (default), the PSDs of all channels is displayed. No averaging
        is done and parameters area_mode and area_alpha are ignored. When
        False, it is possible to paint an area (hold left mouse button and
        drag) to plot a topomap.
    show : bool
        Show figure if True.
    %(n_jobs)s
    line_alpha : float | None
        Alpha for the PSD line. Can be None (default) to use 1.0 when
        ``average=True`` and 0.1 when ``average=False``.
    spatial_colors : bool
        Whether to use spatial colors. Only used when ``average=False``.
    xscale : str
        Can be 'linear' (default) or 'log'.
    reject_by_annotation : bool
        Whether to omit bad segments from the data while computing the
        PSD. If True, annotated segments with a description that starts
        with 'bad' are omitted. Has no effect if ``inst`` is an Epochs or
        Evoked object. Defaults to True.

        .. versionadded:: 0.15.0
    %(verbose)s

    Returns
    -------
    fig : instance of Figure
        Figure with frequency spectra of the data channels.
    """
    from matplotlib.ticker import ScalarFormatter

    if average and spatial_colors:
        raise ValueError('Average and spatial_colors cannot be enabled '
                         'simultaneously.')
    if spatial_colors is None:
        spatial_colors = False if average else True

    fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label = _set_psd_plot_params(raw.info, proj, picks, ax, area_mode)
    del ax
    if line_alpha is None:
        line_alpha = 1.0 if average else 0.75
    line_alpha = float(line_alpha)

    psd_list = list()
    ylabels = list()
    if n_fft is None:
        tmax = raw.times[-1] if not np.isfinite(tmax) else tmax
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
    for ii, picks in enumerate(picks_list):
        ax = ax_list[ii]
        psds, freqs = psd_welch(raw, tmin=tmin, tmax=tmax, picks=picks,
                                fmin=fmin, fmax=fmax, proj=proj, n_fft=n_fft,
                                n_overlap=n_overlap, n_jobs=n_jobs,
                                reject_by_annotation=reject_by_annotation)

        ylabel = _convert_psds(psds, dB, estimate, scalings_list[ii],
                               units_list[ii],
                               [raw.ch_names[pi] for pi in picks])

        if average:
            psd_mean = np.mean(psds, axis=0)
            if area_mode == 'std':
                psd_std = np.std(psds, axis=0)
                hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
            elif area_mode == 'range':
                hyp_limits = (np.min(psds, axis=0), np.max(psds, axis=0))
            else:  # area_mode is None
                hyp_limits = None

            ax.plot(freqs, psd_mean, color=color, alpha=line_alpha,
                    linewidth=0.5)
            if hyp_limits is not None:
                ax.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1],
                                color=color, alpha=area_alpha)
        else:
            psd_list.append(psds)

        if make_label:
            if ii == len(picks_list) - 1:
                ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(ylabel)
            ax.set_title(titles_list[ii])
            ax.set_xlim(freqs[0], freqs[-1])

        ylabels.append(ylabel)

    for key, ls in zip(['lowpass', 'highpass', 'line_freq'],
                       ['--', '--', '-.']):
        if raw.info[key] is not None:
            for ax in ax_list:
                ax.axvline(raw.info[key], color='k', linestyle=ls, alpha=0.25,
                           linewidth=2, zorder=2)

    if not average:
        picks = np.concatenate(picks_list)

        psd_list = np.concatenate(psd_list)
        types = np.array([channel_type(raw.info, idx) for idx in picks])
        # Needed because the data does not match the info anymore.
        info = create_info([raw.ch_names[p] for p in picks], raw.info['sfreq'],
                           types)
        info['chs'] = [raw.info['chs'][p] for p in picks]
        valid_channel_types = ['mag', 'grad', 'eeg', 'seeg', 'eog', 'ecg',
                               'emg', 'dipole', 'gof', 'bio', 'ecog', 'hbo',
                               'hbr', 'misc']
        ch_types_used = list()
        for this_type in valid_channel_types:
            if this_type in types:
                ch_types_used.append(this_type)
        assert len(ch_types_used) == len(ax_list)
        unit = ''
        units = {t: yl for t, yl in zip(ch_types_used, ylabels)}
        titles = {c: t for c, t in zip(ch_types_used, titles_list)}
        picks = np.arange(len(psd_list))
        if not spatial_colors:
            spatial_colors = color
        _plot_lines(psd_list, info, picks, fig, ax_list, spatial_colors,
                    unit, units=units, scalings=None, hline=None, gfp=False,
                    types=types, zorder='std', xlim=(freqs[0], freqs[-1]),
                    ylim=None, times=freqs, bad_ch_idx=[], titles=titles,
                    ch_types_used=ch_types_used, selectable=True, psd=True,
                    line_alpha=line_alpha, nave=None)
    for ax in ax_list:
        ax.grid(True, linestyle=':')
        if xscale == 'log':
            ax.set(xscale='log')
            ax.set(xlim=[freqs[1] if freqs[0] == 0 else freqs[0], freqs[-1]])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
    if make_label:
        tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1, fig=fig)
    plt_show(show)
    return fig


def _prepare_mne_browse_raw(params, title, bgcolor, color, bad_color, inds,
                            n_channels):
    """Set up the mne_browse_raw window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    size = get_config('MNE_BROWSE_RAW_SIZE')
    if size is not None:
        size = size.split(',')
        size = tuple([float(s) for s in size])
        size = tuple([float(s) for s in size])

    fig = figure_nobar(facecolor=bgcolor, figsize=size)
    fig.canvas.set_window_title(title if title else "Raw")
    ax = plt.subplot2grid((10, 10), (0, 1), colspan=8, rowspan=9)
    ax_hscroll = plt.subplot2grid((10, 10), (9, 1), colspan=8)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Time (s)')
    ax_vscroll = plt.subplot2grid((10, 10), (0, 9), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_help_button = plt.subplot2grid((10, 10), (0, 0), colspan=1)
    help_button = mpl.widgets.Button(ax_help_button, 'Help')
    help_button.on_clicked(partial(_onclick_help, params=params))
    # store these so they can be fixed on resize
    params['fig'] = fig
    params['ax'] = ax
    params['ax_hscroll'] = ax_hscroll
    params['ax_vscroll'] = ax_vscroll
    params['ax_help_button'] = ax_help_button
    params['help_button'] = help_button

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
                                       alpha=0.25, linewidth=1, clip_on=False)
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

    # default key to close window
    params['close_key'] = 'escape'


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
    labels = params['ax'].yaxis.get_ticklabels()
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
            # do NOT operate in-place lest this get screwed up
            this_data = params['data'][inds[ch_ind]] * params['scale_factor']
            this_color = bad_color if ch_name in info['bads'] else color

            if isinstance(this_color, dict):
                this_color = this_color[params['types'][inds[ch_ind]]]

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
            vars(lines[ii])['def_color'] = color[params['types'][inds[ch_ind]]]
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
        else:
            # "remove" lines
            lines[ii].set_xdata([])
            lines[ii].set_ydata([])

    params['ax'].texts = []   # delete event and annotation texts

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
