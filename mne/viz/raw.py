"""Functions to plot raw M/EEG data."""
from __future__ import print_function

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

import copy
from functools import partial
from warnings import warn

import numpy as np

from ..externals.six import string_types
from ..io.pick import (pick_types, _pick_data_channels, pick_info,
                       _PICK_TYPES_KEYS, pick_channels, channel_type)
from ..io.proj import setup_proj
from ..io.meas_info import create_info
from ..utils import verbose, get_config
from ..time_frequency import psd_welch
from .topo import _plot_topo, _plot_timeseries, _plot_timeseries_unified
from .utils import (_toggle_options, _toggle_proj, tight_layout,
                    _layout_figure, _plot_raw_onkey, figure_nobar, plt_show,
                    _plot_raw_onscroll, _mouse_click, _find_channel_idx,
                    _helper_raw_resize, _select_bads, _onclick_help,
                    _setup_browser_offsets, _compute_scalings, plot_sensors,
                    _radio_clicked, _set_radio_button, _handle_topomap_bads,
                    _change_channel_group, _plot_annotations)
from ..defaults import _handle_default
from .evoked import _plot_lines


def _plot_update_raw_proj(params, bools):
    """Deal with changed proj."""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
                                        verbose=False)
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
        data[data_picks] = filtfilt(params['ba'][0], params['ba'][1],
                                    data[data_picks], axis=1, padlen=0)
    # scale
    for di in range(data.shape[0]):
        data[di] /= params['scalings'][params['types'][di]]
        # stim channels should be hard limited
        if params['types'][di] == 'stim':
            norm = float(max(data[di]))
            data[di] /= norm if norm > 0 else 1.
    # clip
    if params['clipping'] == 'transparent':
        data[np.logical_or(data > 1, data < -1)] = np.nan
    elif params['clipping'] == 'clamp':
        data = np.clip(data, -1, 1, data)
    params['data'] = data
    params['times'] = times


def _pick_bad_channels(event, params):
    """Selecting / drop bad channels onpick."""
    # Both bad lists are updated. params['info'] used for colors.
    if params['fig_annotation'] is not None:
        return
    bads = params['raw'].info['bads']
    params['info']['bads'] = _select_bads(event, params, bads)
    _plot_update_raw_proj(params, None)


def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order='type',
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
             show_first_samp=False):
    """Plot raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to plot.
    events : array | None
        Events to show with vertical bars.
    duration : float
        Time window (sec) to plot. The lesser of this value and the duration
        of the raw file will be used.
    start : float
        Initial time to show (can be changed dynamically once plotted). If
        show_first_samp is True, then it is taken relative to
        ``raw.first_samp``.
    n_channels : int
        Number of channels to plot at once. Defaults to 20. Has no effect if
        ``order`` is 'position' or 'selection'.
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
                 resp=1, chpi=1e-4)

    remove_dc : bool
        If True remove DC component when plotting data.
    order : str | array of int
        Order in which to plot data. 'type' groups by channel type, 'original'
        plots in the order of ch_names, 'selection' uses Elekta's channel
        groupings (only works for Neuromag data), 'position' groups the
        channels by the positions of the sensors. 'selection' and 'position'
        modes allow custom selections by using lasso selector on the topomap.
        Pressing ``ctrl`` key while selecting allows appending to the current
        selection. If array, only the channels in the array are plotted in the
        given order. Defaults to 'type'.
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
        Filtering order. Note that for efficiency and simplicity,
        filtering during plotting uses forward-backward IIR filtering,
        so the effective filter order will be twice ``filtorder``.
        Filtering the lines for display may also produce some edge
        artifacts (at the left and right edges) of the signals
        during display. Filtering requires scipy >= 0.10.
    clipping : str | None
        If None, channels are allowed to exceed their designated bounds in
        the plot. If "clamp", then values are clamped to the appropriate
        range for display, creating step-like artifacts. If "transparent",
        then excessive values are not shown, creating gaps in the traces.
    show_first_samp : bool
        If True, show time axis relative to the ``raw.first_samp``.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Raw traces.

    Notes
    -----
    The arrow keys (up/down/left/right) can typically be used to navigate
    between channels and time ranges, but this depends on the backend
    matplotlib is configured to use (e.g., mpl.use('TkAgg') should work). The
    scaling can be adjusted with - and + (or =) keys. The viewport dimensions
    can be adjusted with page up/page down and home/end keys. Full screen mode
    can be to toggled with f11 key. To mark or un-mark a channel as bad, click
    on the rather flat segments of a channel's time series. The changes will be
    reflected immediately in the raw object's ``raw.info['bads']`` entry.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.signal import butter
    color = _handle_default('color', color)
    scalings = _compute_scalings(scalings, raw)
    scalings = _handle_default('scalings_plot_raw', scalings)

    if clipping is not None and clipping not in ('clamp', 'transparent'):
        raise ValueError('clipping must be None, "clamp", or "transparent", '
                         'not %s' % clipping)
    # figure out the IIR filtering parameters
    nyq = raw.info['sfreq'] / 2.
    if highpass is None and lowpass is None:
        ba = None
    else:
        filtorder = int(filtorder)
        if filtorder <= 0:
            raise ValueError('filtorder (%s) must be >= 1' % filtorder)
        if highpass is not None and highpass <= 0:
            raise ValueError('highpass must be > 0, not %s' % highpass)
        if lowpass is not None and lowpass >= nyq:
            raise ValueError('lowpass must be < nyquist (%s), not %s'
                             % (nyq, lowpass))
        if highpass is None:
            ba = butter(filtorder, lowpass / nyq, 'lowpass', analog=False)
        elif lowpass is None:
            ba = butter(filtorder, highpass / nyq, 'highpass', analog=False)
        else:
            if lowpass <= highpass:
                raise ValueError('lowpass (%s) must be > highpass (%s)'
                                 % (lowpass, highpass))
            ba = butter(filtorder, [highpass / nyq, lowpass / nyq], 'bandpass',
                        analog=False)

    # make a copy of info, remove projection (for now)
    info = copy.deepcopy(raw.info)
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
    elif not isinstance(title, string_types):
        raise TypeError('title must be None or a string')
    if events is not None:
        event_times = events[:, 0].astype(float) - raw.first_samp
        event_times /= info['sfreq']
        event_nums = events[:, 2]
    else:
        event_times = event_nums = None

    # reorganize the data in plotting order
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

    # put them back to original or modified order for natral plotting
    reord = np.argsort(inds)
    types = [types[ri] for ri in reord]
    if isinstance(order, string_types):
        if order == 'original':
            inds = inds[reord]
        elif order in ['selection', 'position']:
            selections, fig_selection = _setup_browser_selection(raw, order)
        elif order != 'type':
            raise ValueError('Unknown order type %s' % order)
    elif isinstance(order, (np.ndarray, list)):
        # put back to original order first, then use new order
        inds = inds[reord][order]

    if not isinstance(event_color, dict):
        event_color = {-1: event_color}
    else:
        event_color = copy.deepcopy(event_color)  # we might modify it
    for key in event_color:
        if not isinstance(key, int):
            raise TypeError('event_color key "%s" was a %s not an int'
                            % (key, type(key)))
        if key <= 0 and key != -1:
            raise KeyError('only key <= 0 allowed is -1 (cannot use %s)'
                           % key)

    # set up projection and data parameters
    duration = min(raw.times[-1], float(duration))
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc, ba=ba,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, event_times=event_times, inds=inds,
                  event_nums=event_nums, clipping=clipping, fig_proj=None,
                  first_time=first_time, added_label=list())

    if isinstance(order, string_types) and order in ['selection', 'position']:
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

    # do initial plots
    callback_proj('none')
    _layout_figure(params)

    # deal with projectors
    if show_options is True:
        _toggle_options(None, params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    # initialize the first selection set
    if isinstance(order, string_types) and order in ['selection', 'position']:
        _radio_clicked(fig_selection.radio.labels[0]._text, params)
        callback_selection_key = partial(_selection_key_press, params=params)
        callback_selection_scroll = partial(_selection_scroll, params=params)
        params['fig_selection'].canvas.mpl_connect('close_event',
                                                   callback_close)
        params['fig_selection'].canvas.mpl_connect('key_press_event',
                                                   callback_selection_key)
        params['fig_selection'].canvas.mpl_connect('scroll_event',
                                                   callback_selection_scroll)

    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    return params['fig']


def _selection_scroll(event, params):
    """Callback for scroll in selection dialog."""
    if event.step < 0:
        _change_channel_group(-1, params)
    elif event.step > 0:
        _change_channel_group(1, params)


def _selection_key_press(event, params):
    """Callback for keys in selection dialog."""
    if event.key == 'down':
        _change_channel_group(-1, params)
    elif event.key == 'up':
        _change_channel_group(1, params)
    elif event.key == 'escape':
        _close_event(event, params)


def _close_event(event, params):
    """Callback for closing of raw browser with selections."""
    import matplotlib.pyplot as plt
    if 'fig_selection' in params:
        plt.close(params['fig_selection'])
    if 'fig_annotation' is not None:
        plt.close(params['fig_annotation'])
    plt.close(params['fig'])


def _label_clicked(pos, params):
    """Select bad channels."""
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


def _set_psd_plot_params(info, proj, picks, ax, area_mode):
    """Set PSD plot params."""
    import matplotlib.pyplot as plt
    if area_mode not in [None, 'std', 'range']:
        raise ValueError('"area_mode" must be "std", "range", or None')
    if picks is None:
        # XXX this could be refactored more with e.g., plot_evoked
        megs = ['mag', 'grad', False, False, False]
        eegs = [False, False, True, False, False]
        seegs = [False, False, False, True, False]
        ecogs = [False, False, False, False, True]
        names = ['mag', 'grad', 'eeg', 'seeg', 'ecog']
        titles = _handle_default('titles', None)
        units = _handle_default('units', None)
        scalings = _handle_default('scalings', None)
        picks_list = list()
        titles_list = list()
        units_list = list()
        scalings_list = list()
        for meg, eeg, seeg, ecog, name in zip(megs, eegs, seegs, ecogs, names):
            picks = pick_types(info, meg=meg, eeg=eeg, seeg=seeg, ecog=ecog,
                               ref_meg=False)
            if len(picks) > 0:
                picks_list.append(picks)
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
    else:
        picks_list = [picks]
        titles_list = ['Selected channels']
        units_list = ['amplitude']
        scalings_list = [1.]
        ax_list = [ax]

    make_label = False
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

    return (fig, picks_list, titles_list, units_list, scalings_list,
            ax_list, make_label)


def _convert_psds(psds, dB, scaling, unit, ch_names):
    """Convert PSDs to dB (if necessary) and appropriate units."""
    if dB:
        where = np.where(psds.min(1) <= 0)[0]
        if len(where) > 0:
            raise ValueError("Infinite value in PSD for channel(s) %s. "
                             "These channels might be dead." %
                             ', '.join(ch_names[ii] for ii in where))
        psds *= scaling * scaling
        np.log10(psds, out=psds)
        psds *= 10
        ylabel = '%s/Hz (dB)' % unit
    else:
        np.sqrt(psds, out=psds)
        psds *= scaling
        ylabel = '$\\frac{%s}{\\sqrt{Hz}}$' % unit
    return ylabel


@verbose
def plot_raw_psd(raw, tmin=0., tmax=np.inf, fmin=0, fmax=np.inf, proj=False,
                 n_fft=None, picks=None, ax=None, color='black',
                 area_mode='std', area_alpha=0.33, n_overlap=0, dB=True,
                 average=None, show=True, n_jobs=1, line_alpha=None,
                 spatial_colors=None, xscale='linear', verbose=None):
    """Plot the power spectral density across channels.

    Parameters
    ----------
    raw : instance of io.Raw
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
    picks : array-like of int | None
        List of channels to use. Cannot be None if `ax` is supplied. If both
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
        If True, transform data to decibels. If False, plot amplitudes.
    average : bool
        If False, the PSDs of all channels is displayed. No averaging
        is done and parameters area_mode and area_alpha are ignored. When
        False, it is possible to paint an area (hold left mouse button and
        drag) to plot a topomap.
    show : bool
        Show figure if True.
    n_jobs : int
        Number of jobs to run in parallel.
    line_alpha : float | None
        Alpha for the PSD line. Can be None (default) to use 1.0 when
        ``average=True`` and 0.1 when ``average=False``.
    spatial_colors : bool
        Whether to use spatial colors. Only used when ``average=False``.
    xscale : str
        Can be 'linear' (default) or 'log'.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of matplotlib figure
        Figure with frequency spectra of the data channels.
    """
    from matplotlib.ticker import ScalarFormatter
    if average is None:
        warn('In version 0.15 average will default to False and '
             'spatial_colors will default to True.', DeprecationWarning)
        average = True

    if average and spatial_colors:
        raise ValueError('Average and spatial_colors cannot be enabled '
                         'simultaneously.')
    if spatial_colors is None:
        spatial_colors = False if average else True

    fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label = _set_psd_plot_params(raw.info, proj, picks, ax, area_mode)
    del ax
    if line_alpha is None:
        line_alpha = 1.0 if average else 0.1
    line_alpha = float(line_alpha)

    psd_list = list()
    if n_fft is None:
        tmax = raw.times[-1] if not np.isfinite(tmax) else tmax
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
    for ii, picks in enumerate(picks_list):
        ax = ax_list[ii]
        psds, freqs = psd_welch(raw, tmin=tmin, tmax=tmax, picks=picks,
                                fmin=fmin, fmax=fmax, proj=proj, n_fft=n_fft,
                                n_overlap=n_overlap, n_jobs=n_jobs)

        ylabel = _convert_psds(psds, dB, scalings_list[ii], units_list[ii],
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
                ax.set_xlabel('Freq (Hz)')
            ax.set_ylabel(ylabel)
            ax.set_title(titles_list[ii])
            ax.set_xlim(freqs[0], freqs[-1])
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
        unit = 'dB/Hz' if dB else '$1/\\sqrt{Hz}$)'
        units = {t: 'PSD (%s)' % unit for t in ch_types_used}
        titles = {c: t for c, t in zip(ch_types_used, titles_list)}
        picks = np.arange(len(psd_list))
        if not spatial_colors:
            spatial_colors = color
        _plot_lines(psd_list, info, picks, fig, ax_list, spatial_colors,
                    unit, units=units, scalings=None, hline=None, gfp=False,
                    types=types, zorder='std', xlim=(freqs[0], freqs[-1]),
                    ylim=None, times=freqs, bad_ch_idx=[], titles=titles,
                    ch_types_used=ch_types_used, selectable=True, psd=True,
                    line_alpha=line_alpha)
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

    fig = figure_nobar(facecolor=bgcolor, figsize=size)
    fig.canvas.set_window_title('mne_browse_raw')
    ax = plt.subplot2grid((10, 10), (0, 1), colspan=8, rowspan=9)
    ax.set_title(title, fontsize=12)
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
    params['ax_vertline'] = ax.plot([0, 0], ax.get_ylim(),
                                    color=vertline_color, zorder=-1)[0]
    params['ax_vertline'].ch_name = ''
    params['vertline_t'] = ax_hscroll.text(params['first_time'], 1, '',
                                           color=vertline_color,
                                           va='bottom', ha='right')
    params['ax_hscroll_vertline'] = ax_hscroll.plot([0, 0], [0, 1],
                                                    color=vertline_color,
                                                    zorder=2)[0]
    # make shells for plotting traces
    _setup_browser_offsets(params, n_channels)
    ax.set_xlim(params['t_start'], params['t_start'] + params['duration'],
                False)

    params['lines'] = [ax.plot([np.nan], antialiased=False, linewidth=0.5)[0]
                       for _ in range(n_ch)]
    ax.set_yticklabels(['X' * max([len(ch) for ch in info['ch_names']])])
    params['fig_annotation'] = None
    params['segment_line'] = None


def _plot_raw_traces(params, color, bad_color, event_lines=None,
                     event_color=None):
    """Plot raw traces."""
    lines = params['lines']
    info = params['info']
    inds = params['inds']
    n_channels = params['n_channels']
    params['bad_color'] = bad_color
    labels = params['ax'].yaxis.get_ticklabels()
    # do the plotting
    tick_list = list()
    for ii in range(n_channels):
        ch_ind = ii + params['ch_start']
        # let's be generous here and allow users to pass
        # n_channels per view >= the number of traces available
        if ii >= len(lines):
            break
        elif ch_ind < len(inds):
            # scale to fit
            ch_name = info['ch_names'][inds[ch_ind]]
            tick_list += [ch_name]
            offset = params['offsets'][ii]

            # do NOT operate in-place lest this get screwed up
            this_data = params['data'][inds[ch_ind]] * params['scale_factor']
            this_color = bad_color if ch_name in info['bads'] else color
            this_z = 0 if ch_name in info['bads'] else 1
            if isinstance(this_color, dict):
                this_color = this_color[params['types'][inds[ch_ind]]]

            # subtraction here gets corect orientation for flipped ylim
            lines[ii].set_ydata(offset - this_data)
            lines[ii].set_xdata(params['times'] + params['first_time'])
            lines[ii].set_color(this_color)
            lines[ii].set_zorder(this_z)
            vars(lines[ii])['ch_name'] = ch_name
            vars(lines[ii])['def_color'] = color[params['types'][inds[ch_ind]]]

            # set label color
            this_color = bad_color if ch_name in info['bads'] else 'black'
            labels[ii].set_color(this_color)
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
            else:
                line.set_xdata([])
                line.set_ydata([])

    if 'segments' in params:
        while len(params['ax'].collections) > 0:
            params['ax'].collections.pop(0)
            params['ax'].texts.pop(0)
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
            dscr = params['annot_description'][idx]
            segment_color = params['segment_colors'][dscr]
            params['ax'].fill_betweenx(ylim, start, end, color=segment_color,
                                       alpha=0.3)
            params['ax'].text((start + end) / 2., ylim[0], dscr, ha='center')

    # finalize plot
    params['ax'].set_xlim(params['times'][0] + params['first_time'],
                          params['times'][0] + params['first_time'] +
                          params['duration'], False)
    params['ax'].set_yticklabels(tick_list)
    if 'fig_selection' not in params:
        params['vsel_patch'].set_y(params['ch_start'])
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def plot_raw_psd_topo(raw, tmin=0., tmax=None, fmin=0., fmax=100., proj=False,
                      n_fft=2048, n_overlap=0, layout=None, color='w',
                      fig_facecolor='k', axis_facecolor='k', dB=True,
                      show=True, block=False, n_jobs=1, verbose=None):
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
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of matplotlib figure
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
                        times=freqs)
    click_func = partial(_plot_timeseries, data=[psds], color=color,
                         times=freqs)
    picks = _pick_data_channels(raw.info)
    info = pick_info(raw.info, picks)

    fig = _plot_topo(info, times=freqs, show_func=show_func,
                     click_func=click_func, layout=layout,
                     axis_facecolor=axis_facecolor,
                     fig_facecolor=fig_facecolor, x_label='Frequency (Hz)',
                     unified=True, y_label=y_label)

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


def _setup_browser_selection(raw, kind):
    """Organize browser selections."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons
    from ..selection import (read_selection, _SELECTIONS, _EEG_SELECTIONS,
                             _divide_to_regions)
    from ..utils import _get_stim_channel
    if kind in ('position'):
        order = _divide_to_regions(raw.info)
        keys = _SELECTIONS[1:]  # no 'Vertex'
    elif 'selection':
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
