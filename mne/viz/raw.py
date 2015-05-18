"""Functions to plot raw M/EEG data
"""
from __future__ import print_function

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import copy
from functools import partial

import numpy as np

from ..externals.six import string_types
from ..io.pick import pick_types
from ..io.proj import setup_proj
from ..utils import set_config, get_config, verbose, deprecated
from ..time_frequency import compute_raw_psd
from .utils import figure_nobar, _toggle_options, _toggle_proj, tight_layout
from ..defaults import _handle_default


def _plot_update_raw_proj(params, bools):
    """Helper only needs to be called when proj is changed"""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
                                        verbose=False)
    _update_raw_data(params)
    params['plot_fun']()


def _update_raw_data(params):
    """Helper only needs to be called when time or proj is changed"""
    from scipy.signal import filtfilt
    start = params['t_start']
    stop = params['raw'].time_as_index(start + params['duration'])[0]
    start = params['raw'].time_as_index(start)[0]
    data_picks = pick_types(params['raw'].info, meg=True, eeg=True)
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
            data[di] = np.minimum(data[di], 1.0)
    # clip
    if params['clipping'] == 'transparent':
        data[np.logical_or(data > 1, data < -1)] = np.nan
    elif params['clipping'] == 'clamp':
        data = np.clip(data, -1, 1, data)
    params['data'] = data
    params['times'] = times


def _layout_raw(params):
    """Set raw figure layout"""
    s = params['fig'].get_size_inches()
    scroll_width = 0.33
    hscroll_dist = 0.33
    vscroll_dist = 0.1
    l_border = 1.2
    r_border = 0.1
    t_border = 0.33
    b_border = 0.5

    # only bother trying to reset layout if it's reasonable to do so
    if s[0] < 2 * scroll_width or s[1] < 2 * scroll_width + hscroll_dist:
        return

    # convert to relative units
    scroll_width_x = scroll_width / s[0]
    scroll_width_y = scroll_width / s[1]
    vscroll_dist /= s[0]
    hscroll_dist /= s[1]
    l_border /= s[0]
    r_border /= s[0]
    t_border /= s[1]
    b_border /= s[1]
    # main axis (traces)
    ax_width = 1.0 - scroll_width_x - l_border - r_border - vscroll_dist
    ax_y = hscroll_dist + scroll_width_y + b_border
    ax_height = 1.0 - ax_y - t_border
    params['ax'].set_position([l_border, ax_y, ax_width, ax_height])
    # vscroll (channels)
    pos = [ax_width + l_border + vscroll_dist, ax_y,
           scroll_width_x, ax_height]
    params['ax_vscroll'].set_position(pos)
    # hscroll (time)
    pos = [l_border, b_border, ax_width, scroll_width_y]
    params['ax_hscroll'].set_position(pos)
    # options button
    pos = [l_border + ax_width + vscroll_dist, b_border,
           scroll_width_x, scroll_width_y]
    params['ax_button'].set_position(pos)
    params['fig'].canvas.draw()


def _helper_resize(event, params):
    """Helper for resizing"""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size)
    _layout_raw(params)


def _pick_bad_channels(event, params):
    """Helper for selecting / dropping bad channels onpick"""
    bads = params['raw'].info['bads']

    # trade-off, avoid selecting more than one channel when drifts are present
    # however for clean data don't click on peaks but on flat segments
    def f(x, y):
        return y(np.mean(x), x.std() * 2)
    for l in event.inaxes.lines:
        ydata = l.get_ydata()
        if not isinstance(ydata, list) and not np.isnan(ydata).any():
            ymin, ymax = f(ydata, np.subtract), f(ydata, np.add)
            if ymin <= event.ydata <= ymax:
                this_chan = vars(l)['ch_name']
                if this_chan in params['raw'].ch_names:
                    if this_chan not in bads:
                        bads.append(this_chan)
                        l.set_color(params['bad_color'])
                        l.set_zorder(-1)
                    else:
                        bads.pop(bads.index(this_chan))
                        l.set_color(vars(l)['def_color'])
                        l.set_zorder(0)
                    break
    else:
        x = np.array([event.xdata] * 2)
        params['ax_vertline'].set_data(x, np.array(params['ax'].get_ylim()))
        params['ax_hscroll_vertline'].set_data(x, np.array([0., 1.]))
        params['vertline_t'].set_text('%0.3f' % x[0])
    # update deep-copied info to persistently draw bads
    params['info']['bads'] = bads
    _plot_update_raw_proj(params, None)


def _mouse_click(event, params):
    """Vertical select callback"""
    if event.inaxes is None or event.button != 1:
        return
    plot_fun = params['plot_fun']
    # vertical scrollbar changed
    if event.inaxes == params['ax_vscroll']:
        ch_start = max(int(event.ydata) - params['n_channels'] // 2, 0)
        if params['ch_start'] != ch_start:
            params['ch_start'] = ch_start
            plot_fun()
    # horizontal scrollbar changed
    elif event.inaxes == params['ax_hscroll']:
        _plot_raw_time(event.xdata - params['duration'] / 2, params)

    elif event.inaxes == params['ax']:
        _pick_bad_channels(event, params)


def _plot_raw_time(value, params):
    """Deal with changed time value"""
    info = params['info']
    max_times = params['n_times'] / float(info['sfreq']) - params['duration']
    if value > max_times:
        value = params['n_times'] / info['sfreq'] - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        _update_raw_data(params)
        params['plot_fun']()


def _plot_raw_onkey(event, params):
    """Interpret key presses"""
    import matplotlib.pyplot as plt
    # check for initial plot
    if event is None:
        params['plot_fun']()
        return

    # quit event
    if event.key == 'escape':
        plt.close(params['fig'])
        return

    # change plotting params
    ch_changed = False
    if event.key == 'down':
        params['ch_start'] += params['n_channels']
        ch_changed = True
    elif event.key == 'up':
        params['ch_start'] -= params['n_channels']
        ch_changed = True
    elif event.key == 'right':
        _plot_raw_time(params['t_start'] + params['duration'], params)
        return
    elif event.key == 'left':
        _plot_raw_time(params['t_start'] - params['duration'], params)
        return
    elif event.key in ['o', 'p']:
        _toggle_options(None, params)
        return

    # deal with plotting changes
    if ch_changed:
        _channels_changed(params)


def _channels_changed(params):
    if params['ch_start'] >= len(params['info']['ch_names']):
        params['ch_start'] = 0
    elif params['ch_start'] < 0:
        # wrap to end
        rem = len(params['info']['ch_names']) % params['n_channels']
        params['ch_start'] = len(params['info']['ch_names'])
        params['ch_start'] -= rem if rem != 0 else params['n_channels']
    params['plot_fun']()


def _plot_raw_onscroll(event, params):
    """Interpret scroll events"""
    orig_start = params['ch_start']
    if event.step < 0:
        params['ch_start'] = min(params['ch_start'] + params['n_channels'],
                                 len(params['info']['ch_names']) -
                                 params['n_channels'])
    else:  # event.key == 'up':
        params['ch_start'] = max(params['ch_start'] - params['n_channels'], 0)
    if orig_start != params['ch_start']:
        _channels_changed(params)


def _plot_traces(params, inds, color, bad_color, lines, event_lines,
                 event_color, offsets):
    """Helper for plotting raw"""

    info = params['info']
    n_channels = params['n_channels']
    params['bad_color'] = bad_color
    # do the plotting
    tick_list = []
    for ii in range(n_channels):
        ch_ind = ii + params['ch_start']
        # let's be generous here and allow users to pass
        # n_channels per view >= the number of traces available
        if ii >= len(lines):
            break
        elif ch_ind < len(info['ch_names']):
            # scale to fit
            ch_name = info['ch_names'][inds[ch_ind]]
            tick_list += [ch_name]
            offset = offsets[ii]

            # do NOT operate in-place lest this get screwed up
            this_data = params['data'][inds[ch_ind]]
            this_color = bad_color if ch_name in info['bads'] else color
            this_z = -1 if ch_name in info['bads'] else 0
            if isinstance(this_color, dict):
                this_color = this_color[params['types'][inds[ch_ind]]]

            # subtraction here gets corect orientation for flipped ylim
            lines[ii].set_ydata(offset - this_data)
            lines[ii].set_xdata(params['times'])
            lines[ii].set_color(this_color)
            lines[ii].set_zorder(this_z)
            vars(lines[ii])['ch_name'] = ch_name
            vars(lines[ii])['def_color'] = color[params['types'][inds[ch_ind]]]
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
        for ev_num, line in zip(sorted(event_color.keys())[::-1],
                                event_lines[::-1]):
            mask = (event_nums == ev_num) if ev_num >= 0 else ~used
            assert not np.any(used[mask])
            used[mask] = True
            t = event_times[mask]
            if len(t) > 0:
                xs = list()
                ys = list()
                for tt in t:
                    xs += [tt, tt, np.nan]
                    ys += [0, 2 * n_channels + 1, np.nan]
                line.set_xdata(xs)
                line.set_ydata(ys)
            else:
                line.set_xdata([])
                line.set_ydata([])
    # finalize plot
    params['ax'].set_xlim(params['times'][0],
                          params['times'][0] + params['duration'], False)
    params['ax'].set_yticklabels(tick_list)
    params['vsel_patch'].set_y(params['ch_start'])
    params['fig'].canvas.draw()


def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=None,
             bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
             event_color='cyan', scalings=None, remove_dc=True, order='type',
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None):
    """Plot raw data

    Parameters
    ----------
    raw : instance of Raw
        The raw data to plot.
    events : array | None
        Events to show with vertical bars.
    duration : float
        Time window (sec) to plot in a given time.
    start : float
        Initial time to show (can be changed dynamically once plotted).
    n_channels : int
        Number of channels to plot at once.
    bgcolor : color object
        Color of the background.
    color : dict | color object | None
        Color for the data traces. If None, defaults to::

            dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='r',
                 emg='k', ref_meg='steelblue', misc='k', stim='k',
                 resp='k', chpi='k')

    bad_color : color object
        Color to make bad channels.
    event_color : color object | dict
        Color to use for events. Can also be a dict with
        ``{event_number: color}`` pairings. Use ``event_number==-1`` for
        any event numbers in the events list that are not in the dictionary.
    scalings : dict | None
        Scale factors for the traces. If None, defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                 resp=1, chpi=1e-4)

    remove_dc : bool
        If True remove DC component when plotting data.
    order : 'type' | 'original' | array
        Order in which to plot data. 'type' groups by channel type,
        'original' plots in the order of ch_names, array gives the
        indices to use in plotting.
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

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Raw traces.

    Notes
    -----
    The arrow keys (up/down/left/right) can typically be used to navigate
    between channels and time ranges, but this depends on the backend
    matplotlib is configured to use (e.g., mpl.use('TkAgg') should work).
    To mark or un-mark a channel as bad, click on the rather flat segments
    of a channel's time series. The changes will be reflected immediately
    in the raw object's ``raw.info['bads']`` entry.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.signal import butter
    color = _handle_default('color', color)
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
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    for t in ['eeg', 'eog', 'ecg', 'emg', 'ref_meg', 'stim', 'resp',
              'misc', 'chpi', 'syst', 'ias', 'exci']:
        pick_kwargs[t] = True
        inds += [pick_types(raw.info, **pick_kwargs)]
        types += [t] * len(inds[-1])
        pick_kwargs[t] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(info['ch_names']):
        raise RuntimeError('Some channels not classified, please report '
                           'this problem')

    # put them back to original or modified order for natral plotting
    reord = np.argsort(inds)
    types = [types[ri] for ri in reord]
    if isinstance(order, str):
        if order == 'original':
            inds = inds[reord]
        elif order != 'type':
            raise ValueError('Unknown order type %s' % order)
    elif isinstance(order, np.ndarray):
        if not np.array_equal(np.sort(order),
                              np.arange(len(info['ch_names']))):
            raise ValueError('order, if array, must have integers from '
                             '0 to n_channels - 1')
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
    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc, ba=ba,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, event_times=event_times,
                  event_nums=event_nums, clipping=clipping)

    # set up plotting
    size = get_config('MNE_BROWSE_RAW_SIZE')
    if size is not None:
        size = size.split(',')
        size = tuple([float(s) for s in size])
        # have to try/catch when there's no toolbar
    fig = figure_nobar(facecolor=bgcolor, figsize=size)
    fig.canvas.set_window_title('mne_browse_raw')
    ax = plt.subplot2grid((10, 10), (0, 0), colspan=9, rowspan=9)
    ax.set_title(title, fontsize=12)
    ax_hscroll = plt.subplot2grid((10, 10), (9, 0), colspan=9)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Time (s)')
    ax_vscroll = plt.subplot2grid((10, 10), (0, 9), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_button = plt.subplot2grid((10, 10), (9, 9))
    # store these so they can be fixed on resize
    params['fig'] = fig
    params['ax'] = ax
    params['ax_hscroll'] = ax_hscroll
    params['ax_vscroll'] = ax_vscroll
    params['ax_button'] = ax_button

    # populate vertical and horizontal scrollbars
    for ci in range(len(info['ch_names'])):
        this_color = (bad_color if info['ch_names'][inds[ci]] in info['bads']
                      else color)
        if isinstance(this_color, dict):
            this_color = this_color[types[inds[ci]]]
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=this_color,
                                                   edgecolor=this_color))
    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       facecolor='w', edgecolor='w')
    ax_vscroll.add_patch(vsel_patch)
    params['vsel_patch'] = vsel_patch
    hsel_patch = mpl.patches.Rectangle((start, 0), duration, 1, edgecolor='k',
                                       facecolor=(0.75, 0.75, 0.75),
                                       alpha=0.25, linewidth=1, clip_on=False)
    ax_hscroll.add_patch(hsel_patch)
    params['hsel_patch'] = hsel_patch
    ax_hscroll.set_xlim(0, n_times / float(info['sfreq']))
    n_ch = len(info['ch_names'])
    ax_vscroll.set_ylim(n_ch, 0)
    ax_vscroll.set_title('Ch.')

    # make shells for plotting traces
    offsets = np.arange(n_channels) * 2 + 1
    ylim = [n_channels * 2 + 1, 0]
    ax.set_yticks(offsets)
    ax.set_ylim(ylim)
    # plot event_line first so it's in the back
    event_lines = [ax.plot([np.nan], color=event_color[ev_num])[0]
                   for ev_num in sorted(event_color.keys())]
    lines = [ax.plot([np.nan], antialiased=False, linewidth=0.5)[0]
             for _ in range(n_ch)]
    ax.set_yticklabels(['X' * max([len(ch) for ch in info['ch_names']])])
    vertline_color = (0., 0.75, 0.)
    params['ax_vertline'] = ax.plot([0, 0], ylim, color=vertline_color,
                                    zorder=-1)[0]
    params['ax_vertline'].ch_name = ''
    params['vertline_t'] = ax_hscroll.text(0, 0.5, '0.000',
                                           color=vertline_color,
                                           verticalalignment='center',
                                           horizontalalignment='right')
    params['ax_hscroll_vertline'] = ax_hscroll.plot([0, 0], [0, 1],
                                                    color=vertline_color,
                                                    zorder=1)[0]

    params['plot_fun'] = partial(_plot_traces, params=params, inds=inds,
                                 color=color, bad_color=bad_color, lines=lines,
                                 event_lines=event_lines,
                                 event_color=event_color, offsets=offsets)

    # set up callbacks
    opt_button = None
    if len(raw.info['projs']) > 0:
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
    callback_key = partial(_plot_raw_onkey, params=params)
    fig.canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    fig.canvas.mpl_connect('button_press_event', callback_pick)
    callback_resize = partial(_helper_resize, params=params)
    fig.canvas.mpl_connect('resize_event', callback_resize)

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
    _layout_raw(params)

    # deal with projectors
    params['fig_opts'] = None
    if show_options is True:
        _toggle_options(None, params)

    if show:
        try:
            plt.show(block=block)
        except TypeError:  # not all versions have this
            plt.show()

    return fig


def _set_psd_plot_params(info, proj, picks, ax, area_mode):
    """Aux function"""
    import matplotlib.pyplot as plt
    if area_mode not in [None, 'std', 'range']:
        raise ValueError('"area_mode" must be "std", "range", or None')
    if picks is None:
        if ax is not None:
            raise ValueError('If "ax" is not supplied (None), then "picks" '
                             'must also be supplied')
        megs = ['mag', 'grad', False]
        eegs = [False, False, True]
        names = ['Magnetometers', 'Gradiometers', 'EEG']
        picks_list = list()
        titles_list = list()
        for meg, eeg, name in zip(megs, eegs, names):
            picks = pick_types(info, meg=meg, eeg=eeg, ref_meg=False)
            if len(picks) > 0:
                picks_list.append(picks)
                titles_list.append(name)
        if len(picks_list) == 0:
            raise RuntimeError('No MEG or EEG channels found')
    else:
        picks_list = [picks]
        titles_list = ['Selected channels']
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

    return fig, picks_list, titles_list, ax_list, make_label


@verbose
def plot_raw_psd(raw, tmin=0., tmax=np.inf, fmin=0, fmax=np.inf, proj=False,
                 n_fft=2048, picks=None, ax=None, color='black',
                 area_mode='std', area_alpha=0.33,
                 n_overlap=0, dB=True, show=True, n_jobs=1, verbose=None):
    """Plot the power spectral density across channels

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
    n_fft : int
        Number of points to use in Welch FFT calculations.
    picks : array-like of int | None
        List of channels to use. Cannot be None if `ax` is supplied. If both
        `picks` and `ax` are None, separate subplots will be created for
        each standard channel type (`mag`, `grad`, and `eeg`).
    ax : instance of matplotlib Axes | None
        Axes to plot into. If None, axes will be created.
    color : str | tuple
        A matplotlib-compatible color to use.
    area_mode : str | None
        Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
        will be plotted. If 'range', the min and max (across channels) will be
        plotted. Bad channels will be excluded from these calculations.
        If None, no area will be plotted.
    area_alpha : float
        Alpha for the area.
    n_overlap : int
        The number of points of overlap between blocks. The default value
        is 0 (no overlap).
    dB : bool
        If True, transform data to decibels.
    show : bool
        Show figure if True.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    import matplotlib.pyplot as plt
    fig, picks_list, titles_list, ax_list, make_label = _set_psd_plot_params(
        raw.info, proj, picks, ax, area_mode)

    for ii, (picks, title, ax) in enumerate(zip(picks_list, titles_list,
                                                ax_list)):
        psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                      fmin=fmin, fmax=fmax, proj=proj,
                                      n_fft=n_fft, n_overlap=n_overlap,
                                      n_jobs=n_jobs, verbose=None)

        # Convert PSDs to dB
        if dB:
            psds = 10 * np.log10(psds)
            unit = 'dB'
        else:
            unit = 'power'
        psd_mean = np.mean(psds, axis=0)
        if area_mode == 'std':
            psd_std = np.std(psds, axis=0)
            hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
        elif area_mode == 'range':
            hyp_limits = (np.min(psds, axis=0), np.max(psds, axis=0))
        else:  # area_mode is None
            hyp_limits = None

        ax.plot(freqs, psd_mean, color=color)
        if hyp_limits is not None:
            ax.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1],
                            color=color, alpha=area_alpha)
        if make_label:
            if ii == len(picks_list) - 1:
                ax.set_xlabel('Freq (Hz)')
            if ii == len(picks_list) // 2:
                ax.set_ylabel('Power Spectral Density (%s/Hz)' % unit)
            ax.set_title(title)
            ax.set_xlim(freqs[0], freqs[-1])
    if make_label:
        tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1, fig=fig)
    if show is True:
        plt.show()
    return fig


@deprecated("'plot_raw_psds' is deprecated and will be removed in v0.10, "
            "please use 'plot_raw_psd' instead")
def plot_raw_psds(*args, **kwargs):
    plot_raw_psd(*args, **kwargs)
plot_raw_psds.__doc__ = plot_raw_psd.__doc__
