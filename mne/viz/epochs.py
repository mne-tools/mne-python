"""Functions to plot epochs data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

from collections import Counter
from functools import partial
import copy

import numpy as np

from ..utils import verbose, get_config, set_config, logger, warn
from ..io.pick import pick_types, channel_type
from ..io.proj import setup_proj
from ..time_frequency import psd_multitaper
from .utils import (tight_layout, figure_nobar, _toggle_proj, _toggle_options,
                    _layout_figure, _setup_vmin_vmax, _channels_changed,
                    _plot_raw_onscroll, _onclick_help, plt_show,
                    _compute_scalings, DraggableColorbar, _setup_cmap)
from .misc import _handle_event_colors
from ..defaults import _handle_default


def plot_epochs_image(epochs, picks=None, sigma=0., vmin=None,
                      vmax=None, colorbar=True, order=None, show=True,
                      units=None, scalings=None, cmap='RdBu_r',
                      fig=None, axes=None, overlay_times=None):
    """Plot Event Related Potential / Fields image.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    picks : int | array-like of int | None
        The indices of the channels to consider. If None, the first
        five good channels are plotted.
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image. If 0., no smoothing is applied.
    vmin : float
        The min value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    vmax : float
        The max value in the image. The unit is uV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    colorbar : bool
        Display or not a colorbar.
    order : None | array of int | callable
        If not None, order is used to reorder the epochs on the y-axis
        of the image. If it's an array of int it should be of length
        the number of good epochs. If it's a callable the arguments
        passed are the times vector and the data as 2d array
        (data.shape[1] == len(times).
    show : bool
        Show figure if True.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `units=dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting.
        If None, defaults to `scalings=dict(eeg=1e6, grad=1e13, mag=1e15,
        eog=1e6)`.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive'
        Colormap. If tuple, the first value indicates the colormap to use and
        the second value is a boolean defining interactivity. In interactive
        mode the colors are adjustable by clicking and dragging the colorbar
        with left and right mouse button. Left mouse button moves the scale up
        and down and right mouse button adjusts the range. Hitting space bar
        resets the scale. Up and down arrows can be used to change the
        colormap. If 'interactive', translates to ('RdBu_r', True). Defaults to
        'RdBu_r'.
    fig : matplotlib figure | None
        Figure instance to draw the image to. Figure must contain two axes for
        drawing the single trials and evoked responses. If None a new figure is
        created. Defaults to None.
    axes : list of matplotlib axes | None
        List of axes instances to draw the image, erp and colorbar to.
        Must be of length three if colorbar is True (with the last list element
        being the colorbar axes) or two if colorbar is False. If both fig and
        axes are passed an error is raised. Defaults to None.
    overlay_times : array-like, shape (n_epochs,) | None
        If not None the parameter is interpreted as time instants in seconds
        and is added to the image. It is typically useful to display reaction
        times. Note that it is defined with respect to the order
        of epochs such that overlay_times[0] corresponds to epochs[0].

    Returns
    -------
    figs : lists of matplotlib figures
        One figure per channel displayed.
    """
    from scipy import ndimage
    units = _handle_default('units', units)
    scalings = _handle_default('scalings', scalings)

    import matplotlib.pyplot as plt
    if picks is None:
        picks = pick_types(epochs.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')[:5]

    if set(units.keys()) != set(scalings.keys()):
        raise ValueError('Scalings and units must have the same keys.')

    picks = np.atleast_1d(picks)
    if (fig is not None or axes is not None) and len(picks) > 1:
        raise ValueError('Only single pick can be drawn to a figure.')
    if axes is not None:
        if fig is not None:
            raise ValueError('Both figure and axes were passed, please'
                             'decide between the two.')
        from .utils import _validate_if_list_of_axes
        oblig_len = 3 if colorbar else 2
        _validate_if_list_of_axes(axes, obligatory_len=oblig_len)
        ax1, ax2 = axes[:2]
        # if axes were passed - we ignore fig param and get figure from axes
        fig = ax1.get_figure()
        if colorbar:
            ax3 = axes[-1]
    evoked = epochs.average(picks)
    data = epochs.get_data()[:, picks, :]
    n_epochs = len(data)
    data = np.swapaxes(data, 0, 1)
    if sigma > 0.:
        for k in range(len(picks)):
            data[k, :] = ndimage.gaussian_filter1d(
                data[k, :], sigma=sigma, axis=0)

    scale_vmin = True if vmin is None else False
    scale_vmax = True if vmax is None else False
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

    if overlay_times is not None and len(overlay_times) != n_epochs:
        raise ValueError('size of overlay_times parameter (%s) do not '
                         'match the number of epochs (%s).'
                         % (len(overlay_times), n_epochs))

    if overlay_times is not None:
        overlay_times = np.array(overlay_times)
        times_min = np.min(overlay_times)
        times_max = np.max(overlay_times)
        if ((times_min < epochs.tmin) or (times_max > epochs.tmax)):
            warn('Some values in overlay_times fall outside of the epochs '
                 'time interval (between %s s and %s s)'
                 % (epochs.tmin, epochs.tmax))

    figs = list()
    for i, (this_data, idx) in enumerate(zip(data, picks)):
        if fig is None:
            this_fig = plt.figure()
        else:
            this_fig = fig
        figs.append(this_fig)

        ch_type = channel_type(epochs.info, idx)
        if ch_type not in scalings:
            # We know it's not in either scalings or units since keys match
            raise KeyError('%s type not in scalings and units' % ch_type)
        this_data *= scalings[ch_type]

        this_order = order
        if callable(order):
            this_order = order(epochs.times, this_data)

        if this_order is not None and (len(this_order) != len(this_data)):
            raise ValueError('size of order parameter (%s) does not '
                             'match the number of epochs (%s).'
                             % (len(this_order), len(this_data)))

        this_overlay_times = None
        if overlay_times is not None:
            this_overlay_times = overlay_times

        if this_order is not None:
            this_order = np.asarray(this_order)
            this_data = this_data[this_order]
            if this_overlay_times is not None:
                this_overlay_times = this_overlay_times[this_order]

        plt.figure(this_fig.number)
        if axes is None:
            ax1 = plt.subplot2grid((3, 10), (0, 0), colspan=9, rowspan=2)
            ax2 = plt.subplot2grid((3, 10), (2, 0), colspan=9, rowspan=1)
            if colorbar:
                ax3 = plt.subplot2grid((3, 10), (0, 9), colspan=1, rowspan=3)

        this_vmin = vmin * scalings[ch_type] if scale_vmin else vmin
        this_vmax = vmax * scalings[ch_type] if scale_vmax else vmax

        cmap = _setup_cmap(cmap)
        im = ax1.imshow(this_data,
                        extent=[1e3 * epochs.times[0], 1e3 * epochs.times[-1],
                                0, n_epochs],
                        aspect='auto', origin='lower', interpolation='nearest',
                        vmin=this_vmin, vmax=this_vmax, cmap=cmap[0])
        if this_overlay_times is not None:
            ax1.plot(1e3 * this_overlay_times, 0.5 + np.arange(len(this_data)),
                     'k', linewidth=2)
        ax1.set_title(epochs.ch_names[idx])
        ax1.set_ylabel('Epochs')
        ax1.axis('auto')
        ax1.axis('tight')
        ax1.axvline(0, color='m', linewidth=3, linestyle='--')
        evoked_data = scalings[ch_type] * evoked.data[i]
        ax2.plot(1e3 * evoked.times, evoked_data)
        ax2.set_xlabel('Time (ms)')
        ax2.set_xlim([1e3 * evoked.times[0], 1e3 * evoked.times[-1]])
        ax2.set_ylabel(units[ch_type])
        evoked_vmin = min(evoked_data) * 1.1 if scale_vmin else vmin
        evoked_vmax = max(evoked_data) * 1.1 if scale_vmax else vmax
        if scale_vmin or scale_vmax:
            evoked_vmax = max(np.abs([evoked_vmax, evoked_vmin]))
            evoked_vmin = -evoked_vmax
        ax2.set_ylim([evoked_vmin, evoked_vmax])
        ax2.axvline(0, color='m', linewidth=3, linestyle='--')
        if colorbar:
            cbar = plt.colorbar(im, cax=ax3)
            if cmap[1]:
                ax1.CB = DraggableColorbar(cbar, im)
            tight_layout(fig=this_fig)
    plt_show(show)

    return figs


def plot_drop_log(drop_log, threshold=0, n_max_plot=20, subject='Unknown',
                  color=(0.9, 0.9, 0.9), width=0.8, ignore=('IGNORED',),
                  show=True):
    """Show the channel stats based on a drop_log from Epochs.

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
    from ..epochs import _drop_log_stats
    perc = _drop_log_stats(drop_log, ignore)
    scores = Counter([ch for d in drop_log for ch in d if ch not in ignore])
    ch_names = np.array(list(scores.keys()))
    fig = plt.figure()
    if perc < threshold or len(ch_names) == 0:
        plt.text(0, 0, 'No drops')
        return fig
    n_used = 0
    for d in drop_log:  # "d" is the list of drop reasons for each epoch
        if len(d) == 0 or any(ch not in ignore for ch in d):
            n_used += 1  # number of epochs not ignored
    counts = 100 * np.array(list(scores.values()), dtype=float) / n_used
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
    tight_layout(pad=1, fig=fig)
    plt_show(show)
    return fig


def _draw_epochs_axes(epoch_idx, good_ch_idx, bad_ch_idx, data, times, axes,
                      title_str, axes_handler):
    """Handle drawing epochs axes."""
    this = axes_handler[0]
    for ii, data_, ax in zip(epoch_idx, data, axes):
        for l, d in zip(ax.lines, data_[good_ch_idx]):
            l.set_data(times, d)
        if bad_ch_idx is not None:
            bad_lines = [ax.lines[k] for k in bad_ch_idx]
            for l, d in zip(bad_lines, data_[bad_ch_idx]):
                l.set_data(times, d)
        if title_str is not None:
            ax.set_title(title_str % ii, fontsize=12)
        ax.set_ylim(data.min(), data.max())
        ax.set_yticks(list())
        ax.set_xticks(list())
        if vars(ax)[this]['reject'] is True:
            #  memorizing reject
            for l in ax.lines:
                l.set_color((0.8, 0.8, 0.8))
            ax.get_figure().canvas.draw()
        else:
            #  forgetting previous reject
            for k in axes_handler:
                if k == this:
                    continue
                if vars(ax).get(k, {}).get('reject', None) is True:
                    for l in ax.lines[:len(good_ch_idx)]:
                        l.set_color('k')
                    if bad_ch_idx is not None:
                        for l in ax.lines[-len(bad_ch_idx):]:
                            l.set_color('r')
                    ax.get_figure().canvas.draw()
                    break


def _epochs_navigation_onclick(event, params):
    """Handle epochs navigation click."""
    import matplotlib.pyplot as plt
    p = params
    here = None
    if event.inaxes == p['back'].ax:
        here = 1
    elif event.inaxes == p['next'].ax:
        here = -1
    elif event.inaxes == p['reject-quit'].ax:
        if p['reject_idx']:
            p['epochs'].drop(p['reject_idx'])
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
    """Handle epochs axes click."""
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
            for l in ax.lines:
                l.set_color(reject_color)
            here['reject'] = True
    elif here.get('reject', None) is True:
        idx = here['idx']
        if idx in p['reject_idx']:
            p['reject_idx'].pop(p['reject_idx'].index(idx))
            good_lines = [ax.lines[k] for k in p['good_ch_idx']]
            for l in good_lines:
                l.set_color('k')
            if p['bad_ch_idx'] is not None:
                bad_lines = ax.lines[-len(p['bad_ch_idx']):]
                for l in bad_lines:
                    l.set_color('r')
            here['reject'] = False
    ax.get_figure().canvas.draw()


def plot_epochs(epochs, picks=None, scalings=None, n_epochs=20, n_channels=20,
                title=None, events=None, event_colors=None, show=True,
                block=False):
    """Visualize epochs.

    Bad epochs can be marked with a left click on top of the epoch. Bad
    channels can be selected by clicking the channel name on the left side of
    the main axes. Calling this function drops all the selected bad epochs as
    well as bad epochs marked beforehand with rejection parameters.

    Parameters
    ----------

    epochs : instance of Epochs
        The epochs object
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None
    scalings : dict | 'auto' | None
        Scaling factors for the traces. If any fields in scalings are 'auto',
        the scaling factor is set to match the 99.5th percentile of a subset of
        the corresponding data. If scalings == 'auto', all scalings fields are
        set to 'auto'. If any fields are 'auto' and data is not preloaded,
        a subset of epochs up to 100mb will be loaded. If None, defaults to::

            dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                 emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)

    n_epochs : int
        The number of epochs per view. Defaults to 20.
    n_channels : int
        The number of channels per view. Defaults to 20.
    title : str | None
        The title of the window. If None, epochs name will be displayed.
        Defaults to None.
    events : None, array, shape (n_events, 3)
        Events to show with vertical bars. If events are provided, the epoch
        numbers are not shown to prevent overlap. You can toggle epoch
        numbering through options (press 'o' key). You can use
        :func:`mne.viz.plot_events` as a legend for the colors. By default, the
        coloring scheme is the same.

        .. warning::  If the epochs have been resampled, the events no longer
            align with the data.

        .. versionadded:: 0.14.0
    event_colors : None, dict
        Dictionary of event_id value and its associated color. If None,
        colors are automatically drawn from a default list (cycled through if
        number of events longer than list of default colors). Uses the same
        coloring scheme as :func:`mne.viz.plot_events`.

        .. versionadded:: 0.14.0
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
    The arrow keys (up/down/left/right) can be used to navigate between
    channels and epochs and the scaling can be adjusted with - and + (or =)
    keys, but this depends on the backend matplotlib is configured to use
    (e.g., mpl.use(``TkAgg``) should work). Full screen mode can be toggled
    with f11 key. The amount of epochs and channels per view can be adjusted
    with home/end and page down/page up keys. These can also be set through
    options dialog by pressing ``o`` key. ``h`` key plots a histogram of
    peak-to-peak values along with the used rejection thresholds. Butterfly
    plot can be toggled with ``b`` key. Right mouse click adds a vertical line
    to the plot. Click 'help' button at bottom left corner of the plotter to
    view all the options.

    .. versionadded:: 0.10.0
    """
    epochs.drop_bad()
    scalings = _compute_scalings(scalings, epochs)
    scalings = _handle_default('scalings_plot_raw', scalings)

    projs = epochs.info['projs']

    params = {'epochs': epochs,
              'info': copy.deepcopy(epochs.info),
              'bad_color': (0.8, 0.8, 0.8),
              't_start': 0,
              'histogram': None}
    params['label_click_fun'] = partial(_pick_bad_channels, params=params)
    _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks, events=events,
                               event_colors=event_colors)
    _prepare_projectors(params)
    _layout_figure(params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    return params['fig']


@verbose
def plot_epochs_psd(epochs, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                    proj=False, bandwidth=None, adaptive=False, low_bias=True,
                    normalization='length', picks=None, ax=None, color='black',
                    area_mode='std', area_alpha=0.33, dB=True, n_jobs=1,
                    show=True, verbose=None):
    """Plot the power spectral density across epochs.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object
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
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth.
    normalization : str
        Either "full" or "length" (default). If "full", the PSD will
        be normalized by the sampling rate as well as the length of
        the signal (as in nitime).
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
    dB : bool
        If True, transform data to decibels.
    n_jobs : int
        Number of jobs to run in parallel.
    show : bool
        Show figure if True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    from .raw import _set_psd_plot_params, _convert_psds
    fig, picks_list, titles_list, units_list, scalings_list, ax_list, \
        make_label = _set_psd_plot_params(
            epochs.info, proj, picks, ax, area_mode)

    for ii, (picks, title, ax) in enumerate(zip(picks_list, titles_list,
                                                ax_list)):
        psds, freqs = psd_multitaper(epochs, picks=picks, fmin=fmin,
                                     fmax=fmax, tmin=tmin, tmax=tmax,
                                     bandwidth=bandwidth, adaptive=adaptive,
                                     low_bias=low_bias,
                                     normalization=normalization, proj=proj,
                                     n_jobs=n_jobs)

        ylabel = _convert_psds(psds, dB, scalings_list[ii], units_list[ii],
                               [epochs.ch_names[pi] for pi in picks])

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
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(freqs[0], freqs[-1])
    if make_label:
        tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1, fig=fig)
    plt_show(show)
    return fig


def _prepare_mne_browse_epochs(params, projs, n_channels, n_epochs, scalings,
                               title, picks, events=None, event_colors=None,
                               order=None):
    """Set up the mne_browse_epochs window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.colors import colorConverter
    epochs = params['epochs']

    if picks is None:
        picks = _handle_picks(epochs)
    if len(picks) < 1:
        raise RuntimeError('No appropriate channels found. Please'
                           ' check your picks')
    picks = sorted(picks)
    # Reorganize channels
    inds = list()
    types = list()
    for t in ['grad', 'mag']:
        idxs = pick_types(params['info'], meg=t, ref_meg=False, exclude=[])
        if len(idxs) < 1:
            continue
        mask = np.in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [t] * len(inds[-1])
    for t in ['hbo', 'hbr']:
        idxs = pick_types(params['info'], meg=False, ref_meg=False, fnirs=t,
                          exclude=[])
        if len(idxs) < 1:
            continue
        mask = np.in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [t] * len(inds[-1])
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    if order is None:
        order = ['eeg', 'seeg', 'ecog', 'eog', 'ecg', 'emg', 'ref_meg', 'stim',
                 'resp', 'misc', 'chpi', 'syst', 'ias', 'exci']
    for ch_type in order:
        pick_kwargs[ch_type] = True
        idxs = pick_types(params['info'], **pick_kwargs)
        if len(idxs) < 1:
            continue
        mask = np.in1d(idxs, picks, assume_unique=True)
        inds.append(idxs[mask])
        types += [ch_type] * len(inds[-1])
        pick_kwargs[ch_type] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(picks):
        raise RuntimeError('Some channels not classified. Please'
                           ' check your picks')
    ch_names = [params['info']['ch_names'][x] for x in inds]

    # set up plotting
    size = get_config('MNE_BROWSE_RAW_SIZE')
    n_epochs = min(n_epochs, len(epochs.events))
    duration = len(epochs.times) * n_epochs
    n_channels = min(n_channels, len(picks))
    if size is not None:
        size = size.split(',')
        size = tuple(float(s) for s in size)
    if title is None:
        title = epochs.name
        if epochs.name is None or len(title) == 0:
            title = ''
    fig = figure_nobar(facecolor='w', figsize=size, dpi=80)
    fig.canvas.set_window_title('mne_browse_epochs')
    ax = plt.subplot2grid((10, 15), (0, 1), colspan=13, rowspan=9)

    ax.annotate(title, xy=(0.5, 1), xytext=(0, ax.get_ylim()[1] + 15),
                ha='center', va='bottom', size=12, xycoords='axes fraction',
                textcoords='offset points')
    color = _handle_default('color', None)

    ax.axis([0, duration, 0, 200])
    ax2 = ax.twiny()
    ax2.set_zorder(-1)
    ax2.axis([0, duration, 0, 200])
    ax_hscroll = plt.subplot2grid((10, 15), (9, 1), colspan=13)
    ax_hscroll.get_yaxis().set_visible(False)
    ax_hscroll.set_xlabel('Epochs')
    ax_vscroll = plt.subplot2grid((10, 15), (0, 14), rowspan=9)
    ax_vscroll.set_axis_off()
    ax_vscroll.add_patch(mpl.patches.Rectangle((0, 0), 1, len(picks),
                                               facecolor='w', zorder=3))

    ax_help_button = plt.subplot2grid((10, 15), (9, 0), colspan=1)
    help_button = mpl.widgets.Button(ax_help_button, 'Help')
    help_button.on_clicked(partial(_onclick_help, params=params))

    # populate vertical and horizontal scrollbars
    for ci in range(len(picks)):
        if ch_names[ci] in params['info']['bads']:
            this_color = params['bad_color']
        else:
            this_color = color[types[ci]]
        ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                   facecolor=this_color,
                                                   edgecolor=this_color,
                                                   zorder=4))

    vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                       edgecolor='w', facecolor='w', zorder=5)
    ax_vscroll.add_patch(vsel_patch)

    ax_vscroll.set_ylim(len(types), 0)
    ax_vscroll.set_title('Ch.')

    # populate colors list
    type_colors = [colorConverter.to_rgba(color[c]) for c in types]
    colors = list()
    for color_idx in range(len(type_colors)):
        colors.append([type_colors[color_idx]] * len(epochs.events))
    lines = list()
    n_times = len(epochs.times)

    for ch_idx in range(n_channels):
        if len(colors) - 1 < ch_idx:
            break
        lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                            zorder=3, picker=3.)
        ax.add_collection(lc)
        lines.append(lc)

    times = epochs.times
    data = np.zeros((params['info']['nchan'], len(times) * n_epochs))

    ylim = (25., 0.)  # Hardcoded 25 because butterfly has max 5 rows (5*5=25).
    # make shells for plotting traces
    offset = ylim[0] / n_channels
    offsets = np.arange(n_channels) * offset + (offset / 2.)

    times = np.arange(len(times) * len(epochs.events))
    epoch_times = np.arange(0, len(times), n_times)

    ax.set_yticks(offsets)
    ax.set_ylim(ylim)
    ticks = epoch_times + 0.5 * n_times
    ax.set_xticks(ticks)
    ax2.set_xticks(ticks[:n_epochs])
    labels = list(range(1, len(ticks) + 1))  # epoch numbers
    ax.set_xticklabels(labels)
    xlim = epoch_times[-1] + len(epochs.times)
    ax_hscroll.set_xlim(0, xlim)
    vertline_t = ax_hscroll.text(0, 1, '', color='y', va='bottom', ha='right')

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
    text = ax.text(0, 0, 'blank', zorder=3, verticalalignment='baseline',
                   ha='left', fontweight='bold')
    text.set_visible(False)

    epoch_nr = True
    if events is not None:
        event_set = set(events[:, 2])
        event_colors = _handle_event_colors(event_set, event_colors, event_set)
        epoch_nr = False  # epoch number off by default to avoid overlap
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)

    params.update({'fig': fig,
                   'ax': ax,
                   'ax2': ax2,
                   'ax_hscroll': ax_hscroll,
                   'ax_vscroll': ax_vscroll,
                   'vsel_patch': vsel_patch,
                   'hsel_patch': hsel_patch,
                   'lines': lines,
                   'projs': projs,
                   'ch_names': ch_names,
                   'n_channels': n_channels,
                   'n_epochs': n_epochs,
                   'scalings': scalings,
                   'duration': duration,
                   'ch_start': 0,
                   'colors': colors,
                   'def_colors': type_colors,  # don't change at runtime
                   'picks': picks,
                   'bads': np.array(list(), dtype=int),
                   'data': data,
                   'times': times,
                   'epoch_times': epoch_times,
                   'offsets': offsets,
                   'labels': labels,
                   'scale_factor': 1.0,
                   'butterfly_scale': 1.0,
                   'fig_proj': None,
                   'types': np.array(types),
                   'inds': inds,
                   'vert_lines': list(),
                   'vertline_t': vertline_t,
                   'butterfly': False,
                   'text': text,
                   'ax_help_button': ax_help_button,  # needed for positioning
                   'help_button': help_button,  # reference needed for clicks
                   'fig_options': None,
                   'settings': [True, True, epoch_nr, True],
                   'image_plot': None,
                   'events': events,
                   'event_colors': event_colors,
                   'ev_lines': list(),
                   'ev_texts': list()})

    params['plot_fun'] = partial(_plot_traces, params=params)

    # callbacks
    callback_scroll = partial(_plot_onscroll, params=params)
    fig.canvas.mpl_connect('scroll_event', callback_scroll)
    callback_click = partial(_mouse_click, params=params)
    fig.canvas.mpl_connect('button_press_event', callback_click)
    callback_key = partial(_plot_onkey, params=params)
    fig.canvas.mpl_connect('key_press_event', callback_key)
    callback_resize = partial(_resize_event, params=params)
    fig.canvas.mpl_connect('resize_event', callback_resize)
    fig.canvas.mpl_connect('pick_event', partial(_onpick, params=params))
    params['callback_key'] = callback_key

    # Draw event lines for the first time.
    _plot_vert_lines(params)


def _prepare_projectors(params):
    """Set up the projectors for epochs browser."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    epochs = params['epochs']
    projs = params['projs']
    if len(projs) > 0 and not epochs.proj:
        ax_button = plt.subplot2grid((10, 15), (9, 14))
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
        params['opt_button'] = opt_button
        params['ax_button'] = ax_button

    # As here code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_epochs_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    callback_proj('none')


def _plot_traces(params):
    """Plot concatenated epochs."""
    params['text'].set_visible(False)
    ax = params['ax']
    butterfly = params['butterfly']
    if butterfly:
        ch_start = 0
        n_channels = len(params['picks'])
        data = params['data'] * params['butterfly_scale']
    else:
        ch_start = params['ch_start']
        n_channels = params['n_channels']
        data = params['data'] * params['scale_factor']
    offsets = params['offsets']
    lines = params['lines']
    epochs = params['epochs']

    n_times = len(epochs.times)
    tick_list = list()
    start_idx = int(params['t_start'] / n_times)
    end = params['t_start'] + params['duration']
    end_idx = int(end / n_times)
    xlabels = params['labels'][start_idx:]
    event_ids = params['epochs'].events[:, 2]
    params['ax2'].set_xticklabels(event_ids[start_idx:])
    ax.set_xticklabels(xlabels)
    ylabels = ax.yaxis.get_ticklabels()
    # do the plotting
    for line_idx in range(n_channels):
        ch_idx = line_idx + ch_start
        if line_idx >= len(lines):
            break
        elif ch_idx < len(params['ch_names']):
            if butterfly:
                ch_type = params['types'][ch_idx]
                if ch_type == 'grad':
                    offset = offsets[0]
                elif ch_type == 'mag':
                    offset = offsets[1]
                elif ch_type == 'eeg':
                    offset = offsets[2]
                elif ch_type == 'eog':
                    offset = offsets[3]
                elif ch_type == 'ecg':
                    offset = offsets[4]
                else:
                    lines[line_idx].set_segments(list())
            else:
                tick_list += [params['ch_names'][ch_idx]]
                offset = offsets[line_idx]
            this_data = data[ch_idx]

            # subtraction here gets correct orientation for flipped ylim
            ydata = offset - this_data
            xdata = params['times'][:params['duration']]
            num_epochs = np.min([params['n_epochs'], len(epochs.events)])
            segments = np.split(np.array((xdata, ydata)).T, num_epochs)

            ch_name = params['ch_names'][ch_idx]
            if ch_name in params['info']['bads']:
                if not butterfly:
                    this_color = params['bad_color']
                    ylabels[line_idx].set_color(this_color)
                this_color = np.tile((params['bad_color']), (num_epochs, 1))
                for bad_idx in params['bads']:
                    if bad_idx < start_idx or bad_idx > end_idx:
                        continue
                    this_color[bad_idx - start_idx] = (1., 0., 0.)
                lines[line_idx].set_zorder(2)
            else:
                this_color = params['colors'][ch_idx][start_idx:end_idx]
                lines[line_idx].set_zorder(3)
                if not butterfly:
                    ylabels[line_idx].set_color('black')
            lines[line_idx].set_segments(segments)
            lines[line_idx].set_color(this_color)
        else:
            lines[line_idx].set_segments(list())

    # finalize plot
    ax.set_xlim(params['times'][0], params['times'][0] + params['duration'],
                False)
    params['ax2'].set_xlim(params['times'][0],
                           params['times'][0] + params['duration'], False)
    if butterfly:
        factor = -1. / params['butterfly_scale']
        labels = np.empty(20, dtype='S15')
        labels.fill('')
        ticks = ax.get_yticks()
        idx_offset = 1
        if 'grad' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[0]) *
                                               params['scalings']['grad'] *
                                               1e13 * factor)
            idx_offset += 4
        if 'mag' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[1]) *
                                               params['scalings']['mag'] *
                                               1e15 * factor)
            idx_offset += 4
        if 'eeg' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[2]) *
                                               params['scalings']['eeg'] *
                                               1e6 * factor)
            idx_offset += 4
        if 'eog' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[3]) *
                                               params['scalings']['eog'] *
                                               1e6 * factor)
            idx_offset += 4
        if 'ecg' in params['types']:
            labels[idx_offset + 1] = '0.00'
            for idx in [idx_offset, idx_offset + 2]:
                labels[idx] = '{0:.2f}'.format((ticks[idx] - offsets[4]) *
                                               params['scalings']['ecg'] *
                                               1e6 * factor)
        ax.set_yticklabels(labels, fontsize=12, color='black')
    else:
        ax.set_yticklabels(tick_list, fontsize=12)

    if params['events'] is not None:  # vertical lines for events.
        _draw_event_lines(params)

    params['vsel_patch'].set_y(ch_start)
    params['fig'].canvas.draw()
    # XXX This is a hack to make sure this figure gets drawn last
    # so that when matplotlib goes to calculate bounds we don't get a
    # CGContextRef error on the MacOSX backend :(
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _plot_update_epochs_proj(params, bools=None):
    """Deal with proj changed."""
    if bools is not None:
        inds = np.where(bools)[0]
        params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
                                   for ii in inds]
        params['proj_bools'] = bools
    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
                                        verbose=False)

    start = int(params['t_start'] / len(params['epochs'].times))
    n_epochs = params['n_epochs']
    end = start + n_epochs
    data = np.concatenate(params['epochs'][start:end].get_data(), axis=1)
    if params['projector'] is not None:
        data = np.dot(params['projector'], data)
    types = params['types']
    for pick, ind in enumerate(params['inds']):
        params['data'][pick] = data[ind] / params['scalings'][types[pick]]
    params['plot_fun']()


def _handle_picks(epochs):
    """Handle picks."""
    if any('ICA' in k for k in epochs.ch_names):
        picks = pick_types(epochs.info, misc=True, ref_meg=False,
                           exclude=[])
    else:
        picks = pick_types(epochs.info, meg=True, eeg=True, eog=True, ecg=True,
                           seeg=True, ecog=True, ref_meg=False, fnirs=True,
                           exclude=[])
    return picks


def _plot_window(value, params):
    """Deal with horizontal shift of the viewport."""
    max_times = len(params['times']) - params['duration']
    if value > max_times:
        value = len(params['times']) - params['duration']
    if value < 0:
        value = 0
    if params['t_start'] != value:
        params['t_start'] = value
        params['hsel_patch'].set_x(value)
        params['plot_update_proj_callback'](params)


def _plot_vert_lines(params):
    """Plot vertical lines."""
    ax = params['ax']
    while len(ax.lines) > 0:
        ax.lines.pop()
    params['vert_lines'] = list()
    params['ev_lines'] = list()
    params['vertline_t'].set_text('')

    epochs = params['epochs']
    if params['settings'][3]:  # if zeroline visible
        t_zero = np.where(epochs.times == 0.)[0]
        if len(t_zero) == 1:  # not True if tmin > 0
            for event_idx in range(len(epochs.events)):
                pos = [event_idx * len(epochs.times) + t_zero[0],
                       event_idx * len(epochs.times) + t_zero[0]]
                ax.plot(pos, ax.get_ylim(), 'g', zorder=4, alpha=0.4)
    for epoch_idx in range(len(epochs.events)):
        pos = [epoch_idx * len(epochs.times), epoch_idx * len(epochs.times)]
        ax.plot(pos, ax.get_ylim(), color='black', linestyle='--', zorder=2)
    if params['events'] is not None:
        _draw_event_lines(params)


def _pick_bad_epochs(event, params):
    """Select / drop bad epochs."""
    if 'ica' in params:
        pos = (event.xdata, event.ydata)
        _pick_bad_channels(pos, params)
        return
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
        params['ax_hscroll'].patches[epoch_idx].set_zorder(2)
        params['plot_fun']()
        return
    # add bad epoch
    params['bads'] = np.append(params['bads'], epoch_idx)
    params['ax_hscroll'].patches[epoch_idx].set_color((1., 0., 0., 1.))
    params['ax_hscroll'].patches[epoch_idx].set_zorder(3)
    params['ax_hscroll'].patches[epoch_idx].set_edgecolor('w')
    for ch_idx in range(len(params['ch_names'])):
        params['colors'][ch_idx][epoch_idx] = (1., 0., 0., 1.)
    params['plot_fun']()


def _pick_bad_channels(pos, params):
    """Select bad channels."""
    text, ch_idx = _label2idx(params, pos)
    if text is None:
        return
    if text in params['info']['bads']:
        while text in params['info']['bads']:
            params['info']['bads'].remove(text)
        color = params['def_colors'][ch_idx]
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    else:
        params['info']['bads'].append(text)
        color = params['bad_color']
        params['ax_vscroll'].patches[ch_idx + 1].set_color(color)
    if 'ica' in params:
        params['plot_fun']()
    else:
        params['plot_update_proj_callback'](params)


def _plot_onscroll(event, params):
    """Handle scroll events."""
    if event.key == 'control':
        if event.step < 0:
            event.key = '-'
        else:
            event.key = '+'
        _plot_onkey(event, params)
        return
    if params['butterfly']:
        return
    _plot_raw_onscroll(event, params, len(params['ch_names']))


def _mouse_click(event, params):
    """Handle mouse click events."""
    if event.inaxes is None:
        if params['butterfly'] or not params['settings'][0]:
            return
        ax = params['ax']
        ylim = ax.get_ylim()
        pos = ax.transData.inverted().transform((event.x, event.y))
        if pos[0] > 0 or pos[1] < 0 or pos[1] > ylim[0]:
            return
        if event.button == 1:  # left click
            params['label_click_fun'](pos)
        elif event.button == 3:  # right click
            if 'ica' not in params:
                _, ch_idx = _label2idx(params, pos)
                if ch_idx is None:
                    return
                if channel_type(params['info'], ch_idx) not in ['mag', 'grad',
                                                                'eeg', 'eog']:
                    logger.info('Event related fields / potentials only '
                                'available for MEG and EEG channels.')
                    return
                fig = plot_epochs_image(params['epochs'],
                                        picks=params['inds'][ch_idx],
                                        fig=params['image_plot'])[0]
                params['image_plot'] = fig
    elif event.button == 1:  # left click
        # vertical scroll bar changed
        if event.inaxes == params['ax_vscroll']:
            if params['butterfly']:
                return
            # Don't let scrollbar go outside vertical scrollbar limits
            # XXX: floating point exception on some machines if this happens.
            ch_start = min(
                max(int(event.ydata) - params['n_channels'] // 2, 0),
                len(params['ch_names']) - params['n_channels'])

            if params['ch_start'] != ch_start:
                params['ch_start'] = ch_start
                params['plot_fun']()
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

    elif event.inaxes == params['ax'] and event.button == 2:  # middle click
        params['fig'].canvas.draw()
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    elif event.inaxes == params['ax'] and event.button == 3:  # right click
        n_times = len(params['epochs'].times)
        xdata = int(event.xdata % n_times)
        prev_xdata = 0
        if len(params['vert_lines']) > 0:
            prev_xdata = params['vert_lines'][0][0].get_data()[0][0]
            while len(params['vert_lines']) > 0:
                params['ax'].lines.remove(params['vert_lines'][0][0])
                params['vert_lines'].pop(0)
        if prev_xdata == xdata:  # lines removed
            params['vertline_t'].set_text('')
            params['plot_fun']()
            return
        ylim = params['ax'].get_ylim()
        for epoch_idx in range(params['n_epochs']):  # plot lines
            pos = [epoch_idx * n_times + xdata, epoch_idx * n_times + xdata]
            params['vert_lines'].append(params['ax'].plot(pos, ylim, 'y',
                                                          zorder=5))
        params['vertline_t'].set_text('%0.3f' % params['epochs'].times[xdata])
        params['plot_fun']()


def _plot_onkey(event, params):
    """Handle key presses."""
    import matplotlib.pyplot as plt
    if event.key == 'down':
        if params['butterfly']:
            return
        params['ch_start'] += params['n_channels']
        _channels_changed(params, len(params['ch_names']))
    elif event.key == 'up':
        if params['butterfly']:
            return
        params['ch_start'] -= params['n_channels']
        _channels_changed(params, len(params['ch_names']))
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
    elif event.key == '-':
        if params['butterfly']:
            params['butterfly_scale'] /= 1.1
        else:
            params['scale_factor'] /= 1.1
        params['plot_fun']()
    elif event.key in ['+', '=']:
        if params['butterfly']:
            params['butterfly_scale'] *= 1.1
        else:
            params['scale_factor'] *= 1.1
        params['plot_fun']()
    elif event.key == 'f11':
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    elif event.key == 'pagedown':
        if params['n_channels'] == 1 or params['butterfly']:
            return
        n_channels = params['n_channels'] - 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        params['ax'].collections.pop()
        params['ax'].set_yticks(params['offsets'])
        params['lines'].pop()
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'pageup':
        if params['butterfly']:
            return
        from matplotlib.collections import LineCollection
        n_channels = params['n_channels'] + 1
        ylim = params['ax'].get_ylim()
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['n_channels'] = n_channels
        lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                            zorder=3, picker=3.)
        params['ax'].add_collection(lc)
        params['ax'].set_yticks(params['offsets'])
        params['lines'].append(lc)
        params['vsel_patch'].set_height(n_channels)
        params['plot_fun']()
    elif event.key == 'home':
        n_epochs = params['n_epochs'] - 1
        if n_epochs <= 0:
            return
        n_times = len(params['epochs'].times)
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        params['duration'] -= n_times
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = params['data'][:, :-n_times]
        params['plot_update_proj_callback'](params)
    elif event.key == 'end':
        n_epochs = params['n_epochs'] + 1
        n_times = len(params['epochs'].times)
        if n_times * n_epochs > len(params['times']):
            return
        ticks = params['epoch_times'] + 0.5 * n_times
        params['ax2'].set_xticks(ticks[:n_epochs])
        params['n_epochs'] = n_epochs
        if len(params['vert_lines']) > 0:
            ax = params['ax']
            pos = params['vert_lines'][0][0].get_data()[0] + params['duration']
            params['vert_lines'].append(ax.plot(pos, ax.get_ylim(), 'y',
                                                zorder=4))
        params['duration'] += n_times
        if params['t_start'] + params['duration'] > len(params['times']):
            params['t_start'] -= n_times
            params['hsel_patch'].set_x(params['t_start'])
        params['hsel_patch'].set_width(params['duration'])
        params['data'] = np.zeros((len(params['data']), params['duration']))
        params['plot_update_proj_callback'](params)
    elif event.key == 'b':
        if params['fig_options'] is not None:
            plt.close(params['fig_options'])
            params['fig_options'] = None
        _prepare_butterfly(params)
        _plot_traces(params)
    elif event.key == 'o':
        if not params['butterfly']:
            _open_options(params)
    elif event.key == 'h':
        _plot_histogram(params)
    elif event.key == '?':
        _onclick_help(event, params)
    elif event.key == 'escape':
        plt.close(params['fig'])


def _prepare_butterfly(params):
    """Set up butterfly plot."""
    from matplotlib.collections import LineCollection
    butterfly = not params['butterfly']
    if butterfly:
        types = set(['grad', 'mag', 'eeg', 'eog',
                     'ecg']) & set(params['types'])
        if len(types) < 1:
            return
        params['ax_vscroll'].set_visible(False)
        ax = params['ax']
        labels = ax.yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(True)
        ylim = (5. * len(types), 0.)
        ax.set_ylim(ylim)
        offset = ylim[0] / (4. * len(types))
        ticks = np.arange(0, ylim[0], offset)
        ticks = [ticks[x] if x < len(ticks) else 0 for x in range(20)]
        ax.set_yticks(ticks)
        used_types = 0
        params['offsets'] = [ticks[2]]
        if 'grad' in types:
            pos = (0, 1 - (ticks[2] / ylim[0]))
            params['ax2'].annotate('Grad (fT/cm)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'mag' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('Mag (fT)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'eeg' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('EEG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'eog' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('EOG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1
        params['offsets'].append(ticks[2 + used_types * 4])
        if 'ecg' in types:
            pos = (0, 1 - (ticks[2 + used_types * 4] / ylim[0]))
            params['ax2'].annotate('ECG (uV)', xy=pos, xytext=(-70, 0),
                                   ha='left', size=12, va='center',
                                   xycoords='axes fraction', rotation=90,
                                   textcoords='offset points')
            used_types += 1

        while len(params['lines']) < len(params['picks']):
            lc = LineCollection(list(), antialiased=False, linewidths=0.5,
                                zorder=3, picker=3.)
            ax.add_collection(lc)
            params['lines'].append(lc)
    else:  # change back to default view
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
        params['ax_vscroll'].set_visible(True)
        while len(params['ax2'].texts) > 0:
            params['ax2'].texts.pop()
        n_channels = params['n_channels']
        while len(params['lines']) > n_channels:
            params['ax'].collections.pop()
            params['lines'].pop()
        ylim = (25., 0.)
        params['ax'].set_ylim(ylim)
        offset = ylim[0] / n_channels
        params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
        params['ax'].set_yticks(params['offsets'])
    params['butterfly'] = butterfly


def _onpick(event, params):
    """Add a channel name on click."""
    if event.mouseevent.button != 2 or not params['butterfly']:
        return  # text label added with a middle mouse button
    lidx = np.where([l is event.artist for l in params['lines']])[0][0]
    text = params['text']
    text.set_x(event.mouseevent.xdata)
    text.set_y(event.mouseevent.ydata)
    text.set_text(params['ch_names'][lidx])
    text.set_visible(True)
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use _mouse_click (happens once per click)
    # to do the drawing


def _close_event(event, params):
    """Drop selected bad epochs (called on closing of the plot)."""
    params['epochs'].drop(params['bads'])
    params['epochs'].info['bads'] = params['info']['bads']
    logger.info('Channels marked as bad: %s' % params['epochs'].info['bads'])


def _resize_event(event, params):
    """Handle resize event."""
    size = ','.join([str(s) for s in params['fig'].get_size_inches()])
    set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
    _layout_figure(params)


def _update_channels_epochs(event, params):
    """Change the amount of channels and epochs per view."""
    from matplotlib.collections import LineCollection
    # Channels
    n_channels = int(np.around(params['channel_slider'].val))
    offset = params['ax'].get_ylim()[0] / n_channels
    params['offsets'] = np.arange(n_channels) * offset + (offset / 2.)
    while len(params['lines']) > n_channels:
        params['ax'].collections.pop()
        params['lines'].pop()
    while len(params['lines']) < n_channels:
        lc = LineCollection(list(), linewidths=0.5, antialiased=False,
                            zorder=3, picker=3.)
        params['ax'].add_collection(lc)
        params['lines'].append(lc)
    params['ax'].set_yticks(params['offsets'])
    params['vsel_patch'].set_height(n_channels)
    params['n_channels'] = n_channels

    # Epochs
    n_epochs = int(np.around(params['epoch_slider'].val))
    n_times = len(params['epochs'].times)
    ticks = params['epoch_times'] + 0.5 * n_times
    params['ax2'].set_xticks(ticks[:n_epochs])
    params['n_epochs'] = n_epochs
    params['duration'] = n_times * n_epochs
    params['hsel_patch'].set_width(params['duration'])
    params['data'] = np.zeros((len(params['data']), params['duration']))
    if params['t_start'] + n_times * n_epochs > len(params['times']):
        params['t_start'] = len(params['times']) - n_times * n_epochs
        params['hsel_patch'].set_x(params['t_start'])
    params['plot_update_proj_callback'](params)


def _toggle_labels(label, params):
    """Toggle axis labels."""
    if label == 'Channel names visible':
        params['settings'][0] = not params['settings'][0]
        labels = params['ax'].yaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][0])
    elif label == 'Event-id visible':
        params['settings'][1] = not params['settings'][1]
        labels = params['ax2'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][1])
    elif label == 'Epoch-id visible':
        params['settings'][2] = not params['settings'][2]
        labels = params['ax'].xaxis.get_ticklabels()
        for label in labels:
            label.set_visible(params['settings'][2])
    elif label == 'Zeroline visible':
        params['settings'][3] = not params['settings'][3]
        _plot_vert_lines(params)
    params['fig'].canvas.draw()
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _open_options(params):
    """Open the option window."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if params['fig_options'] is not None:
        # turn off options dialog
        plt.close(params['fig_options'])
        params['fig_options'] = None
        return
    width = 10
    height = 3
    fig_options = figure_nobar(figsize=(width, height), dpi=80)
    fig_options.canvas.set_window_title('View settings')
    params['fig_options'] = fig_options
    ax_channels = plt.axes([0.15, 0.1, 0.65, 0.1])
    ax_epochs = plt.axes([0.15, 0.25, 0.65, 0.1])
    ax_button = plt.axes([0.85, 0.1, 0.1, 0.25])
    ax_check = plt.axes([0.15, 0.4, 0.4, 0.55])
    plt.axis('off')
    params['update_button'] = mpl.widgets.Button(ax_button, 'Update')
    params['channel_slider'] = mpl.widgets.Slider(ax_channels, 'Channels', 1,
                                                  len(params['ch_names']),
                                                  valfmt='%0.0f',
                                                  valinit=params['n_channels'])
    params['epoch_slider'] = mpl.widgets.Slider(ax_epochs, 'Epochs', 1,
                                                len(params['epoch_times']),
                                                valfmt='%0.0f',
                                                valinit=params['n_epochs'])
    params['checkbox'] = mpl.widgets.CheckButtons(ax_check,
                                                  ['Channel names visible',
                                                   'Event-id visible',
                                                   'Epoch-id visible',
                                                   'Zeroline visible'],
                                                  actives=params['settings'])
    update = partial(_update_channels_epochs, params=params)
    params['update_button'].on_clicked(update)
    labels_callback = partial(_toggle_labels, params=params)
    params['checkbox'].on_clicked(labels_callback)
    close_callback = partial(_settings_closed, params=params)
    params['fig_options'].canvas.mpl_connect('close_event', close_callback)
    try:
        params['fig_options'].canvas.draw()
        params['fig_options'].show(warn=False)
        if params['fig_proj'] is not None:
            params['fig_proj'].canvas.draw()
    except Exception:
        pass


def _settings_closed(events, params):
    """Handle close event from settings dialog."""
    params['fig_options'] = None


def _plot_histogram(params):
    """Plott histogram of peak-to-peak values."""
    import matplotlib.pyplot as plt
    epochs = params['epochs']
    p2p = np.ptp(epochs.get_data(), axis=2)
    types = list()
    data = list()
    if 'eeg' in params['types']:
        eegs = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'eeg'])
        data.append(eegs.ravel())
        types.append('eeg')
    if 'mag' in params['types']:
        mags = np.array([p2p.T[i] for i,
                         x in enumerate(params['types']) if x == 'mag'])
        data.append(mags.ravel())
        types.append('mag')
    if 'grad' in params['types']:
        grads = np.array([p2p.T[i] for i,
                          x in enumerate(params['types']) if x == 'grad'])
        data.append(grads.ravel())
        types.append('grad')
    params['histogram'] = plt.figure()
    scalings = _handle_default('scalings')
    units = _handle_default('units')
    titles = _handle_default('titles')
    colors = _handle_default('color')
    for idx in range(len(types)):
        ax = plt.subplot(len(types), 1, idx + 1)
        plt.xlabel(units[types[idx]])
        plt.ylabel('count')
        color = colors[types[idx]]
        rej = None
        if epochs.reject is not None and types[idx] in epochs.reject.keys():
                rej = epochs.reject[types[idx]] * scalings[types[idx]]
                rng = [0., rej * 1.1]
        else:
            rng = None
        plt.hist(data[idx] * scalings[types[idx]], bins=100, color=color,
                 range=rng)
        if rej is not None:
            ax.plot((rej, rej), (0, ax.get_ylim()[1]), color='r')
        plt.title(titles[types[idx]])
    params['histogram'].suptitle('Peak-to-peak histogram', y=0.99)
    params['histogram'].subplots_adjust(hspace=0.6)
    try:
        params['histogram'].show(warn=False)
    except:
        pass
    if params['fig_proj'] is not None:
        params['fig_proj'].canvas.draw()


def _label2idx(params, pos):
    """Handle click on labels (returns channel name and idx)."""
    labels = params['ax'].yaxis.get_ticklabels()
    offsets = np.array(params['offsets']) + params['offsets'][0]
    line_idx = np.searchsorted(offsets, pos[1])
    text = labels[line_idx].get_text()
    if len(text) == 0:
        return None, None
    ch_idx = params['ch_start'] + line_idx
    return text, ch_idx


def _draw_event_lines(params):
    """Function for drawing event lines."""
    epochs = params['epochs']
    n_times = len(epochs.times)
    start_idx = int(params['t_start'] / n_times)
    color = params['event_colors']
    ax = params['ax']
    for ev_line in params['ev_lines']:
        ax.lines.remove(ev_line)  # clear the view first
    for ev_text in params['ev_texts']:
        ax.texts.remove(ev_text)
    params['ev_texts'] = list()
    params['ev_lines'] = list()
    t_zero = np.where(epochs.times == 0.)[0]  # idx of 0s
    if len(t_zero) == 0:
        t_zero = epochs.times[0] * -1 * epochs.info['sfreq']  # if tmin > 0
    end = params['n_epochs'] + start_idx
    samp_times = params['events'][:, 0]
    for idx, event in enumerate(epochs.events[start_idx:end]):
        event_mask = ((event[0] - t_zero < samp_times) &
                      (samp_times < event[0] + n_times - t_zero))
        for ev in params['events'][event_mask]:
            if ev[0] == event[0]:  # don't redraw the zeroline
                continue
            pos = [idx * n_times + ev[0] - event[0] + t_zero,
                   idx * n_times + ev[0] - event[0] + t_zero]
            kwargs = {} if ev[2] not in color else {'color': color[ev[2]]}
            params['ev_lines'].append(ax.plot(pos, ax.get_ylim(),
                                              zorder=3, **kwargs)[0])
            params['ev_texts'].append(ax.text(pos[0], ax.get_ylim()[0],
                                              ev[2], color=color[ev[2]],
                                              ha='center', va='top'))
