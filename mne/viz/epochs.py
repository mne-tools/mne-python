"""Functions to plot epochs data
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

from collections import deque
from functools import partial
import copy
from sys import platform

import numpy as np

from ..utils import create_chunks, verbose, get_config, deprecated
from ..io.pick import pick_types, channel_type
from ..io.proj import setup_proj
from ..fixes import Counter, _in1d
from ..time_frequency import compute_epochs_psd
from .utils import tight_layout, _prepare_trellis, figure_nobar
from .utils import _toggle_options, _toggle_proj, _layout_figure
from ..defaults import _handle_default


def plot_image_epochs(epochs, picks=None, sigma=0.3, vmin=None,
                      vmax=None, colorbar=True, order=None, show=True,
                      units=None, scalings=None, cmap='RdBu_r'):
    """Plot Event Related Potential / Fields image

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    picks : int | array-like of int | None
        The indices of the channels to consider. If None, all good
        data channels are plotted.
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image.
    vmin : float
        The min value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers
    vmax : float
        The max value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers
    colorbar : bool
        Display or not a colorbar
    order : None | array of int | callable
        If not None, order is used to reorder the epochs on the y-axis
        of the image. If it's an array of int it should be of length
        the number of good epochs. If it's a callable the arguments
        passed are the times vector and the data as 2d array
        (data.shape[1] == len(times)
    show : bool
        Show figure if True.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `units=dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting.
        If None, defaults to `scalings=dict(eeg=1e6, grad=1e13, mag=1e15)`
    cmap : matplotlib colormap
        Colormap.

    Returns
    -------
    figs : the list of matplotlib figures
        One figure per channel displayed
    """
    from scipy import ndimage
    units = _handle_default('units', units)
    scalings = _handle_default('scalings', scalings)

    import matplotlib.pyplot as plt
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

    if set(units.keys()) != set(scalings.keys()):
        raise ValueError('Scalings and units must have the same keys.')

    picks = np.atleast_1d(picks)
    evoked = epochs.average(picks)
    data = epochs.get_data()[:, picks, :]
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    figs = list()
    for i, (this_data, idx) in enumerate(zip(np.swapaxes(data, 0, 1), picks)):
        this_fig = plt.figure()
        figs.append(this_fig)

        ch_type = channel_type(epochs.info, idx)
        if ch_type not in scalings:
            # We know it's not in either scalings or units since keys match
            raise KeyError('%s type not in scalings and units' % ch_type)
        this_data *= scalings[ch_type]

        this_order = order
        if callable(order):
            this_order = order(epochs.times, this_data)

        if this_order is not None:
            this_data = this_data[this_order]

        this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

        ax1 = plt.subplot2grid((3, 10), (0, 0), colspan=9, rowspan=2)
        im = plt.imshow(this_data,
                        extent=[1e3 * epochs.times[0], 1e3 * epochs.times[-1],
                                0, len(data)],
                        aspect='auto', origin='lower',
                        vmin=vmin, vmax=vmax, cmap=cmap)
        ax2 = plt.subplot2grid((3, 10), (2, 0), colspan=9, rowspan=1)
        if colorbar:
            ax3 = plt.subplot2grid((3, 10), (0, 9), colspan=1, rowspan=3)
        ax1.set_title(epochs.ch_names[idx])
        ax1.set_ylabel('Epochs')
        ax1.axis('auto')
        ax1.axis('tight')
        ax1.axvline(0, color='m', linewidth=3, linestyle='--')
        ax2.plot(1e3 * evoked.times, scalings[ch_type] * evoked.data[i])
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel(units[ch_type])
        ax2.set_ylim([vmin, vmax])
        ax2.axvline(0, color='m', linewidth=3, linestyle='--')
        if colorbar:
            plt.colorbar(im, cax=ax3)
            tight_layout(fig=this_fig)

    if show:
        plt.show()

    return figs


def _drop_log_stats(drop_log, ignore=['IGNORED']):
    """
    Parameters
    ----------
    drop_log : list of lists
        Epoch drop log from Epochs.drop_log.
    ignore : list
        The drop reasons to ignore.

    Returns
    -------
    perc : float
        Total percentage of epochs dropped.
    """
    # XXX: This function should be moved to epochs.py after
    # removal of perc return parameter in plot_drop_log()

    if not isinstance(drop_log, list) or not isinstance(drop_log[0], list):
        raise ValueError('drop_log must be a list of lists')

    perc = 100 * np.mean([len(d) > 0 for d in drop_log
                          if not any(r in ignore for r in d)])

    return perc


def plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject='Unknown',
                  color=(0.9, 0.9, 0.9), width=0.8, ignore=['IGNORED'],
                  show=True):
    """Show the channel stats based on a drop_log from Epochs

    Parameters
    ----------
    drop_log : list of lists
        Epoch drop log from Epochs.drop_log.
    threshold : float
        The percentage threshold to use to decide whether or not to
        plot. Default is zero (always plot).
    n_max_plot : int
        Maximum number of channels to show stats for.
    subject : str
        The subject name to use in the title of the plot.
    color : tuple | str
        Color to use for the bars.
    width : float
        Width of the bars.
    ignore : list
        The drop reasons to ignore.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    perc = _drop_log_stats(drop_log, ignore)
    scores = Counter([ch for d in drop_log for ch in d if ch not in ignore])
    ch_names = np.array(list(scores.keys()))
    fig = plt.figure()
    if perc < threshold or len(ch_names) == 0:
        plt.text(0, 0, 'No drops')
        return fig
    counts = 100 * np.array(list(scores.values()), dtype=float) / len(drop_log)
    n_plot = min(n_max_plot, len(ch_names))
    order = np.flipud(np.argsort(counts))
    plt.title('%s: %0.1f%%' % (subject, perc))
    x = np.arange(n_plot)
    plt.bar(x, counts[order[:n_plot]], color=color, width=width)
    plt.xticks(x + width / 2.0, ch_names[order[:n_plot]], rotation=45,
               horizontalalignment='right')
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.ylabel('% of epochs rejected')
    plt.xlim((-width / 2.0, (n_plot - 1) + width * 3 / 2))
    plt.grid(True, axis='y')

    if show:
        plt.show()

    return fig


def _draw_epochs_axes(epoch_idx, good_ch_idx, bad_ch_idx, data, times, axes,
                      title_str, axes_handler):
    """Aux functioin"""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        [l.set_data(times, d) for l, d in zip(ax.lines, data_[good_ch_idx])]
        if bad_ch_idx is not None:
            bad_lines = [ax.lines[k] for k in bad_ch_idx]
            [l.set_data(times, d) for l, d in zip(bad_lines,
                                                  data_[bad_ch_idx])]
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks(list())
        ax.set_xticks(list())
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            [l.set_color((0.8, 0.8, 0.8)) for l in ax.lines]
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    [l.set_color('k') for l in ax.lines[:len(good_ch_idx)]]
                    if bad_ch_idx is not None:
                        [l.set_color('r') for l in ax.lines[-len(bad_ch_idx):]]
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Aux function"""
    import matplotlib.pyplot as plt
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            p['epochs'].drop_epochs(p['reject_idx'])
        plt.close(p['fig'])
        plt.close(event.inaxes.get_figure())

    if here is not None:
        p['idx_handler'].rotate(here)
        p['axes_handler'].rotate(here)
        this_idx = p['idx_handler'][0]
        _draw_epochs_axes(this_idx, p['good_ch_idx'], p['bad_ch_idx'],
                          p['data'][this_idx],
                          p['times'], p['axes'], p['title_str'],
                          p['axes_handler'])
        # XXX don't ask me why
        p['axes'][0].get_figure().canvas.draw()


def _epochs_axes_onclick(event, params):
    """Aux function"""
    reject_color = (0.8, 0.8, 0.8)
    ax = event.inaxes
    if event.inaxes is None:
        return
    p = params
    here = vars(ax)[p['axes_handler'][0]]
    if here.get('reject', None) is False:
        idx = here['idx']
        if idx not in p['reject_idx']:
            p['reject_idx'].append(idx)
            [l.set_color(reject_color) for l in ax.lines]
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            [l.set_color('k') for l in good_lines]
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                [l.set_color('r') for l in bad_lines]
            here['reject'] = False
    ax.get_figure().canvas.draw()


@deprecated("It will be removed in version 0.11. Use trellis=False "
            "option in epochs.plot method.")
def plot_epochs_trellis(epochs, epoch_idx=None, picks=None, scalings=None,
                        title_str='#%003i', show=True, block=False,
                        n_epochs=20):
    """ Visualize epochs using Trellis plot.

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs object
    epoch_idx : array-like | int | None
        The epochs to visualize. If None, the first 20 epochs are shown.
        Defaults to None.
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    scalings : dict | None
        Scale factors for the traces. If None, defaults to:
        `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4, emg=1e-3,
             ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
    title_str : None | str
        The string formatting to use for axes titles. If None, no titles
        will be shown. Defaults expand to ``#001, #002, ...``
    show : bool
        Show figure if True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on a
        sub plot. Defaults to False.
    n_epochs : int
        The number of epochs per view. Defaults to 20.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    scalings = _handle_default('scalings_plot_raw', scalings)
    if np.isscalar(epoch_idx):
        epoch_idx = [epoch_idx]
    if epoch_idx is None:
        n_events = len(epochs.events)
        epoch_idx = list(range(n_events))
    else:
        n_events = len(epoch_idx)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, n_epochs))

    if picks is None:
        picks = _handle_picks(epochs)
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    times = epochs.times * 1e3
    n_channels = epochs.info['nchan']
    types = [channel_type(epochs.info, idx) for idx in
             picks]

    # preallocation needed for min / max scaling
    data = np.zeros((len(epochs.events), n_channels, len(times)))
    for ii, epoch in enumerate(epochs.get_data()):
        for jj, (this_type, this_channel) in enumerate(zip(types, epoch)):
            data[ii, jj] = this_channel / scalings[this_type]

    n_events = len(epochs.events)
    epoch_idx = epoch_idx[:n_events]
    idx_handler = deque(create_chunks(epoch_idx, n_epochs))
    # handle bads
    bad_ch_idx = None
    ch_names = epochs.ch_names
    bads = epochs.info['bads']
    if any(ch_names[k] in bads for k in picks):
        ch_picked = [k for k in ch_names if ch_names.index(k) in picks]
        bad_ch_idx = [ch_picked.index(k) for k in bads if k in ch_names]
        good_ch_idx = [p for p in picks if p not in bad_ch_idx]
    else:
        good_ch_idx = np.arange(n_channels)

    fig, axes = _prepare_trellis(len(data[idx_handler[0]]), max_col=5)
    axes_handler = deque(list(range(len(idx_handler))))
    for ii, data_, ax in zip(idx_handler[0], data[idx_handler[0]], axes):
        ax.plot(times, data_[good_ch_idx].T, color='k')
        if bad_ch_idx is not None:
            ax.plot(times, data_[bad_ch_idx].T, color='r')
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks([])
        ax.set_xticks([])
        vars(ax)[axes_handler[0]] = {'idx': ii, 'reject': False}

    # initialize memory
    for this_view, this_inds in zip(axes_handler, idx_handler):
        for ii, ax in zip(this_inds, axes):
            vars(ax)[this_view] = {'idx': ii, 'reject': False}

    tight_layout(fig=fig)
    navigation = figure_nobar(figsize=(3, 1.5))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])

    params = {
        'fig': fig,
        'idx_handler': idx_handler,
        'epochs': epochs,
        'picks': picks,
        'times': times,
        'scalings': scalings,
        'good_ch_idx': good_ch_idx,
        'bad_ch_idx': bad_ch_idx,
        'axes': axes,
        'back': mpl.widgets.Button(ax1, 'back'),
        'next': mpl.widgets.Button(ax2, 'next'),
        'reject-quit': mpl.widgets.Button(ax3, 'reject-quit'),
        'title_str': title_str,
        'reject_idx': [],
        'axes_handler': axes_handler,
        'data': data,
        'navigation': navigation,
    }
    fig.canvas.mpl_connect('button_press_event',
                           partial(_epochs_axes_onclick, params=params))
    navigation.canvas.mpl_connect('button_press_event',
                                  partial(_epochs_navigation_onclick,
                                          params=params))
    if show:
        plt.show(block=block)
    return fig


def plot_epochs(epochs, picks=None, scalings=None, n_epochs=20,
                n_channels=10, fig_title=None, show=True, block=False):
    """ Visualize epochs.

    Bad epochs can be marked with a left click on top of the epoch.
    Calling this function drops all the selected bad epochs as well as bad
    epochs marked beforehand with rejection parameters.

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs object
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    scalings : dict | None
        Scale factors for the traces. If None, defaults to:
        `dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4, emg=1e-3,
             ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)`
    n_epochs : int
        The number of epochs per view. Defaults to 20.
    n_channels : int
        The number of channels per view. Defaults to 10.
    fig_title : str | None
        The title of the window. If None, epochs name will be displayed.
        Defaults to None.
    show : bool
        Show figure if True. Defaults to True
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for rejecting bad trials on the fly by clicking on an epoch.
        Defaults to False.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.

    Notes
    -----
    With trellis set to False, the arrow keys (up/down/left/right) can
    be used to navigate between channels and epochs and the scaling can be
    adjusted with 'page up' and 'page down' keys, but this depends on the
    backend matplotlib is configured to use
    (e.g., mpl.use(``TkAgg``) should work).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.colors import colorConverter
    scalings = _handle_default('scalings_plot_raw', scalings)
    color = _handle_default('color', None)
    bad_color = (1.0, 0.0, 0.0)

    epoch_data = epochs.get_data()
    if picks is None:
        picks = _handle_picks(epochs)
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    n_epochs = min(n_epochs, len(epochs.events))
    duration = len(epochs.times) * n_epochs
    n_channels = min(n_channels, len(picks))
    times = epochs.times * 1e3
    projs = epochs.info['projs']

    # Reorganize channels
    inds = list()
    types = list()
    for t in ['grad', 'mag']:
        idxs = pick_types(epochs.info, meg=t, ref_meg=False, exclude=[])
        if len(idxs) < 1:
            continue
        mask = _in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [t] * len(inds[-1])
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    for ch_type in ['eeg', 'eog', 'ecg', 'emg', 'ref_meg', 'stim', 'resp',
                    'misc', 'chpi', 'syst', 'ias', 'exci']:
        pick_kwargs[ch_type] = True
        idxs = pick_types(epochs.info, **pick_kwargs)
        if len(idxs) < 1:
            continue
        mask = _in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [ch_type] * len(inds[-1])
        pick_kwargs[ch_type] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(picks):
        raise RuntimeError('Some channels not classified. Please'
                           ' check your picks')

    data = np.zeros((len(epochs.events), epochs.info['nchan'], len(times)))
    for ii, epoch in enumerate(epoch_data):
        for pick, ind in enumerate(inds):
            data[ii, pick] = epoch[ind] / scalings[types[pick]]

    # set up plotting
    size = get_config('MNE_BROWSE_RAW_SIZE')
    if size is not None:
        size = size.split(',')
        size = tuple(float(s) for s in size)
    if fig_title is None:
        fig_title = epochs.name
        if epochs.name is None or len(fig_title) == 0:
            fig_title = ''
    fig = figure_nobar(facecolor='w', figsize=size)
    fig.canvas.set_window_title('mne_browse_epochs')
    ax = plt.subplot2grid((10, 15), (0, 1), colspan=13, rowspan=9)
    ax.set_title(fig_title, fontsize=12)
    ax.axis([0, duration, 0, 200])

    ax_hscroll = plt.subplot2grid((10, 15), (9, 1), colspan=13)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Epochs')
    ax_vscroll = plt.subplot2grid((10, 15), (0, 14), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_button = plt.subplot2grid((10, 15), (9, 14))

    ax_vscroll.add_patch(mpl.patches.Rectangle((0, 0), 1, len(picks),
                                               facecolor='w', zorder=2))
    # populate vertical and horizontal scrollbars
    for ci in range(len(picks)):
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=color[types[ci]],
                                                   edgecolor=color[types[ci]],
                                                   zorder=3))

    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       edgecolor='w', facecolor='w', zorder=4)
    ax_vscroll.add_patch(vsel_patch)

    ax_vscroll.set_ylim(len(types), 0)
    ax_vscroll.set_title('Ch.')

    # populate colors list
    typecolors = [colorConverter.to_rgba(color[c]) for c in types]
    colors = list()
    for ii in range(len(typecolors)):
        colors.append(list())
        colors[ii] = [typecolors[ii]] * len(epochs.events)
    lines = list()
    n_times = len(epochs.times)

    for ch_idx in range(n_channels):
        if len(colors) - 1 < ch_idx:
            break
        lc = LineCollection(list(), linewidths=0.5, colors=colors[ch_idx],
                            zorder=3)
        ax.add_collection(lc)
        lines.append(lc)

    # concatenation
    epoch_data = np.concatenate(data, axis=1)

    ylim = [n_channels * 2.0 + 1, 0]
    # make shells for plotting traces
    offset = ylim[0] / n_channels
    offsets = np.arange(n_channels) * offset + (offset / 2.)

    for epoch_idx in range(n_epochs):
        if epoch_idx % 2 == 1:  # every second area painted yellow
            ax.fill_betweenx(ylim, epoch_idx * n_times,
                             epoch_idx * n_times + n_times, alpha=0.2,
                             facecolor='y', zorder=1)
    times = np.arange(len(data) * len(epochs.times))
    epoch_times = np.arange(0, len(times), n_times)

    ax.set_yticks(offsets)
    ax.set_ylim(ylim)
    ticks = epoch_times + 0.5 * n_times
    ax.set_xticks(ticks)
    labels = list(range(1, len(ticks) + 1))  # epoch numbers
    ax.set_xticklabels(labels)
    xlim = epoch_times[-1] + len(epochs.times)
    ax_hscroll.set_xlim(0, xlim)

    # fit horizontal scroll bar ticks
    hscroll_ticks = np.arange(0, xlim, xlim / 7.0)
    hscroll_ticks = np.append(hscroll_ticks, epoch_times[-1])
    hticks = list()
    for tick in hscroll_ticks:
        hticks.append(epoch_times.flat[np.abs(epoch_times - tick).argmin()])
    hlabels = [x / n_times + 1 for x in hticks]
    ax_hscroll.set_xticks(hticks)
    ax_hscroll.set_xticklabels(hlabels)

    for epoch_idx in range(len(epoch_times)):
        ax_hscroll.add_patch(mpl.patches.Rectangle((epoch_idx * n_times, 0),
                                                   n_times, 1, facecolor='w',
                                                   edgecolor='w', alpha=0.6))
    hsel_patch = mpl.patches.Rectangle((0, 0), duration, 1,
                                       edgecolor='k',
                                       facecolor=(0.75, 0.75, 0.75),
                                       alpha=0.25, linewidth=1, clip_on=False)
    ax_hscroll.add_patch(hsel_patch)

    # color for each event
    ev_colors = plt.cm.rainbow(np.linspace(0, 1, len(epochs.event_id)))
    color_map = dict()
    patches = list()
    used_events = list()
    for idx, event_id in enumerate(set(epochs.events[:, 2])):
        color_map[event_id] = ev_colors[idx]
        if event_id not in used_events:
            patches.append(mpl.patches.Patch(color=ev_colors[idx],
                                             label=str(event_id)))
            used_events += event_id

    params = {'vsel_patch': vsel_patch,
              'epochs': epochs,
              'info': copy.deepcopy(epochs.info),  # needed for projs
              'lines': lines,
              'n_channels': n_channels,
              'n_epochs': n_epochs,
              'fig': fig,
              'ax': ax,
              'ax_hscroll': ax_hscroll,
              'ax_vscroll': ax_vscroll,
              'ax_button': ax_button,
              'ch_start': 0,
              't_start': 0,
              'duration': duration,
              'scalings': scalings,
              'types': types,
              'colors': colors,
              'def_colors': typecolors,  # don't change at runtime
              'picks': picks,
              'bad_color': bad_color,
              'bads': np.array(list(), dtype=int),
              'ch_names': [epochs.info['ch_names'][x] for x in inds],
              'hsel_patch': hsel_patch,
              'data': epoch_data,
              'orig_data': copy.deepcopy(epoch_data),
              'times': times,
              'epoch_times': epoch_times,
              'offsets': offsets,
              'labels': labels,
              'projs': projs,
              'scale_factor': 1.0,
              'fig_proj': None,
              'ev_cmap': color_map}

    legend = ax.legend(handles=patches)
    # Apparently does not work on all platforms.
    if legend is not None:
        ax_legend_button = plt.subplot2grid((10, 15), (9, 0))
        params['ax_legend_button'] = ax_legend_button
        legend_button = mpl.widgets.Button(ax_legend_button, 'Legend')
        legend.set_visible(False)
        params['legend'] = legend
        callback_legend = partial(_toggle_legend, params=params)
        legend_button.on_clicked(callback_legend)

    # callbacks
    opt_button = mpl.widgets.Button(ax_button, 'Proj')
    if len(projs) > 0 and not epochs.proj:
        callback_option = partial(_toggle_options, params=params)
    else:
        callback_option = partial(_show_proj_message)
    opt_button.on_clicked(callback_option)
    callback_scroll = partial(_plot_onscroll, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_scroll)
    callback_click = partial(_mouse_click, params=params)
    fig.canvas.mpl_connect('button_press_event', callback_click)
    callback_key = partial(_plot_onkey, params=params)
    fig.canvas.mpl_connect('key_press_event', callback_key)
    callback_close = partial(_close_event, params=params)
    fig.canvas.mpl_connect('close_event', callback_close)
    callback_resize = partial(_resize_event, params=params)
    fig.canvas.mpl_connect('resize_event', callback_resize)

    # Draw event lines for the first time.
    _plot_events(params)
    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_epochs_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    params['callback_key'] = callback_key

    callback_proj('none')
    _layout_figure(params)

    if show:
        try:
            plt.show(block=block)
        except TypeError:  # not all versions have this
            plt.show()

    return fig


@verbose
def plot_epochs_psd(epochs, fmin=0, fmax=np.inf, proj=False, n_fft=256,
                    picks=None, ax=None, color='black', area_mode='std',
                    area_alpha=0.33, n_overlap=0,
                    dB=True, n_jobs=1, show=True, verbose=None):
    """Plot the power spectral density across epochs

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    proj : bool
        Apply projection.
    n_fft : int
        Number of points to use in Welch FFT calculations.
    picks : array-like of int | None
        List of channels to use.
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
        The number of points of overlap between blocks.
    dB : bool
        If True, transform data to decibels.
    n_jobs : int
        Number of jobs to run in parallel.
    show : bool
        Show figure if True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    import matplotlib.pyplot as plt
    from .raw import _set_psd_plot_params
    fig, picks_list, titles_list, ax_list, make_label = _set_psd_plot_params(
        epochs.info, proj, picks, ax, area_mode)

    for ii, (picks, title, ax) in enumerate(zip(picks_list, titles_list,
                                                ax_list)):
        psds, freqs = compute_epochs_psd(epochs, picks=picks, fmin=fmin,
                                         fmax=fmax, n_fft=n_fft,
                                         n_overlap=n_overlap, proj=proj,
                                         n_jobs=n_jobs)

        # Convert PSDs to dB
        if dB:
            psds = 10 * np.log10(psds)
            unit = 'dB'
        else:
            unit = 'power'
        # mean across epochs and channels
        psd_mean = np.mean(psds, axis=0).mean(axis=0)
        if area_mode == 'std':
            # std across channels
            psd_std = np.std(np.mean(psds, axis=0), axis=0)
            hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
        elif area_mode == 'range':
            hyp_limits = (np.min(np.mean(psds, axis=0), axis=0),
                          np.max(np.mean(psds, axis=0), axis=0))
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
    if show:
        plt.show()
    return fig


def _plot_traces(params):
    """ Helper for plotting concatenated epochs """
    ax = params['ax']
    n_channels = params['n_channels']
    lines = params['lines']
    data = params['data'] * params['scale_factor']
    offsets = params['offsets']
    epochs = params['epochs']

    n_times = len(epochs.times)
    tick_list = list()
    start_idx = int(params['t_start'] / n_times)
    end = params['t_start'] + params['duration']
    end_idx = int(end / n_times)
    labels = params['labels'][start_idx:]
    ax.set_xticklabels(labels)
    # do the plotting
    for ii in range(n_channels):
        ch_ind = ii + params['ch_start']
        if ii >= len(lines):
            break
        elif ch_ind < len(params['picks']):
            ch_name = params['ch_names'][ch_ind]
            tick_list += [ch_name]
            offset = offsets[ii]
            this_data = data[ch_ind][params['t_start']:end]

            # subtraction here gets correct orientation for flipped ylim
            ydata = offset - this_data
            xdata = params['times'][:params['duration']]
            num_epochs = np.min([params['n_epochs'],
                                len(epochs.events)])
            segments = np.split(np.array((xdata, ydata)).T, num_epochs)

            lines[ii].set_segments(segments)
            vars(lines[ii])['ch_name'] = ch_name
            this_color = params['colors'][ch_ind][start_idx:end_idx]
            lines[ii].set_color(this_color)
        else:
            lines[ii].set_segments(list())

    # finalize plot
    ax.set_xlim(params['times'][0], params['times'][0] + params['duration'],
                False)
    ax.set_yticklabels(tick_list)
    params['vsel_patch'].set_y(params['ch_start'])
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _plot_update_epochs_proj(params, bools):
    """Helper only needs to be called when proj is changed"""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
                                        verbose=False)

    data = params['orig_data']
    if params['projector'] is not None:
        params['data'] = np.dot(params['projector'], data)

    _plot_traces(params)


def _handle_picks(epochs):
    """Aux function to handle picks."""
    if any('ICA' in k for k in epochs.ch_names):
            picks = pick_types(epochs.info, misc=True, ref_meg=False,
                               exclude=[])
    else:
        picks = pick_types(epochs.info, meg=True, eeg=True, eog=True,
                           ref_meg=False, exclude=[])
    return picks


def _plot_window(value, params):
    """Deal with horizontal epoch window."""
    max_times = len(params['times']) - params['duration']
    if value > max_times:
        value = len(params['times']) - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        _plot_events(params)
        _plot_traces(params)


def _channels_changed(params):
    """Deal with vertical shift of the viewport."""
    if params['ch_start'] + params['n_channels'] > len(params['ch_names']):
        params['ch_start'] = len(params['ch_names']) - params['n_channels']
    elif params['ch_start'] < 0:
        params['ch_start'] = 0
    _plot_traces(params)


def _plot_events(params):
    """ Helper function for plotting vertical event lines."""
    ax = params['ax']
    ax.lines = list()
    epochs = params['epochs']
    events = epochs.events
    t_zero = np.where(epochs.times == 0.)[0]
    n_times = len(epochs.times)
    start_idx = int(params['t_start'] / n_times)
    end_idx = int((params['t_start'] + params['duration']) / n_times)
    if len(t_zero) == 1:
        for event_idx, event in enumerate(events[start_idx:end_idx]):
            color = params['ev_cmap'][event[2]]
            pos = [event_idx * n_times + t_zero[0],
                   event_idx * n_times + t_zero[0]]
            ax.plot(pos, ax.get_ylim(), color=color, zorder=-1, linewidth=1)


def _pick_bad_epochs(event, params):
    """Helper for selecting / dropping bad epochs"""
    n_times = len(params['epochs'].times)
    start_idx = int(params['t_start'] / n_times)
    xdata = event.xdata
    xlim = event.inaxes.get_xlim()
    epoch_idx = start_idx + int(xdata / (xlim[1] / params['n_epochs']))
    total_epochs = len(params['epochs'].events)
    if epoch_idx > total_epochs - 1:
        return
    # remove bad epoch
    if epoch_idx in params['bads']:
        params['bads'] = params['bads'][(params['bads'] != epoch_idx)]
        for ch_idx in range(len(params['ch_names'])):
            params['colors'][ch_idx][epoch_idx] = params['def_colors'][ch_idx]
        params['ax_hscroll'].patches[epoch_idx].set_color('w')
        params['ax_hscroll'].patches[epoch_idx].set_zorder(1)
        _plot_traces(params)
        return
    # add bad epoch
    params['bads'] = np.append(params['bads'], epoch_idx)
    params['ax_hscroll'].patches[epoch_idx].set_color(params['bad_color'])
    params['ax_hscroll'].patches[epoch_idx].set_zorder(2)
    params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')
    for ch_idx in range(len(params['ch_names'])):
        params['colors'][ch_idx][epoch_idx] = params['bad_color']
    _plot_traces(params)


def _plot_onscroll(event, params):
    """Function to handle scroll events."""
    orig_start = params['ch_start']
    if event.step < 0:
        params['ch_start'] = min(params['ch_start'] + params['n_channels'],
                                 len(params['ch_names']) -
                                 params['n_channels'])
    else:  # event.key == 'up':
        params['ch_start'] = max(params['ch_start'] - params['n_channels'], 0)
    if orig_start != params['ch_start']:
        _channels_changed(params)


def _mouse_click(event, params):
    """Function to handle mouse click events."""
    if event.inaxes is None or event.button != 1:
        return
    # vertical scroll bar changed
    if event.inaxes == params['ax_vscroll']:
        ch_start = max(int(event.ydata) - params['n_channels'] // 2, 0)
        if params['ch_start'] != ch_start:
            params['ch_start'] = ch_start
            _plot_traces(params)
    # horizontal scroll bar changed
    elif event.inaxes == params['ax_hscroll']:
        # find the closest epoch time
        times = params['epoch_times']
        offset = 0.5 * params['n_epochs'] * len(params['epochs'].times)
        xdata = times.flat[np.abs(times - (event.xdata - offset)).argmin()]
        _plot_window(xdata, params)
    # main axes
    elif event.inaxes == params['ax']:
        _pick_bad_epochs(event, params)


def _plot_onkey(event, params):
    """Function to handle key presses."""
    if event.key == 'down':
        params['ch_start'] += params['n_channels']
        _channels_changed(params)
    elif event.key == 'up':
        params['ch_start'] -= params['n_channels']
        _channels_changed(params)
    elif event.key == 'left':
        sample = params['t_start'] - params['duration']
        sample = np.max([0, sample])
        _plot_window(sample, params)
    elif event.key == 'right':
        sample = params['t_start'] + params['duration']
        sample = np.min([sample, params['times'][-1] - params['duration']])
        times = params['epoch_times']
        xdata = times.flat[np.abs(times - sample).argmin()]
        _plot_window(xdata, params)
    elif event.key == 'pagedown':
        params['scale_factor'] /= 1.1
        _plot_traces(params)
    elif event.key == 'pageup':
        params['scale_factor'] *= 1.1
        _plot_traces(params)


def _show_proj_message(event):
    """Function for showing message of applied projectors."""
    import matplotlib.pyplot as plt
    fig_projs = figure_nobar(figsize=(4, 1.5))
    fig_projs.canvas.set_window_title('No projectors.')
    plt.figtext(0.2, 0.4, 'No projection vectors exist\n'
                'or projectors are already applied.')
    fig_projs.show()


def _toggle_legend(event, params):
    """Function for toggling legend on/off."""
    params['legend'].set_visible(not params['legend'].get_visible())
    params['fig'].canvas.draw()


def _close_event(event, params):
    """Function to drop selected bad epochs. Called on closing of the plot."""
    params['epochs'].drop_epochs(params['bads'])


def _resize_event(event, params):
    """Function to handle resize event"""
    _layout_figure(params, platform)
