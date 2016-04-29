"""Functions to make simple plot on evoked M/EEG data (besides topographies)
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import numpy as np

from ..io.pick import (channel_type, pick_types, _picks_by_type,
                       _pick_data_channels)
from ..externals.six import string_types
from ..defaults import _handle_default
from .utils import (_draw_proj_checkbox, tight_layout, _check_delayed_ssp,
                    plt_show, _process_times)
from ..utils import logger, _clean_names, warn
from ..fixes import partial
from ..io.pick import pick_info
from .topo import _plot_evoked_topo
from .topomap import (_prepare_topo_plot, plot_topomap, _check_outlines,
                      _draw_outlines, _prepare_topomap, _topomap_animation)
from ..channels import find_layout


def _butterfly_onpick(event, params):
    """Helper to add a channel name on click"""
    params['need_draw'] = True
    ax = event.artist.get_axes()
    ax_idx = np.where([ax is a for a in params['axes']])[0]
    if len(ax_idx) == 0:  # this can happen if ax param is used
        return  # let the other axes handle it
    else:
        ax_idx = ax_idx[0]
    lidx = np.where([l is event.artist for l in params['lines'][ax_idx]])[0][0]
    ch_name = params['ch_names'][params['idxs'][ax_idx][lidx]]
    text = params['texts'][ax_idx]
    x = event.artist.get_xdata()[event.ind[0]]
    y = event.artist.get_ydata()[event.ind[0]]
    text.set_x(x)
    text.set_y(y)
    text.set_text(ch_name)
    text.set_color(event.artist.get_color())
    text.set_alpha(1.)
    text.set_path_effects(params['path_effects'])
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use on_button_press (happens once per click)
    # to do the drawing


def _butterfly_on_button_press(event, params):
    """Helper to only draw once for picking"""
    if params['need_draw']:
        event.canvas.draw()
    else:
        idx = np.where([event.inaxes is ax for ax in params['axes']])[0]
        if len(idx) == 1:
            text = params['texts'][idx[0]]
            text.set_alpha(0.)
            text.set_path_effects([])
            event.canvas.draw()
    params['need_draw'] = False


def _butterfly_onselect(xmin, xmax, ch_types, evoked, text=None):
    """Function for drawing topomaps from the selected area."""
    import matplotlib.pyplot as plt
    ch_types = [type for type in ch_types if type in ('eeg', 'grad', 'mag')]
    vert_lines = list()
    if text is not None:
        text.set_visible(True)
        ax = text.axes
        ylim = ax.get_ylim()
        vert_lines.append(ax.plot([xmin, xmin], ylim, zorder=0, color='red'))
        vert_lines.append(ax.plot([xmax, xmax], ylim, zorder=0, color='red'))
        fill = ax.fill_betweenx(ylim, x1=xmin, x2=xmax, alpha=0.2,
                                color='green')
        evoked_fig = plt.gcf()
        evoked_fig.canvas.draw()
        evoked_fig.canvas.flush_events()
    times = evoked.times
    xmin *= 0.001
    minidx = np.abs(times - xmin).argmin()
    xmax *= 0.001
    maxidx = np.abs(times - xmax).argmin()
    fig, axarr = plt.subplots(1, len(ch_types), squeeze=False,
                              figsize=(3 * len(ch_types), 3))
    for idx, ch_type in enumerate(ch_types):
        picks, pos, merge_grads, _, ch_type = _prepare_topo_plot(evoked,
                                                                 ch_type,
                                                                 layout=None)
        data = evoked.data[picks, minidx:maxidx]
        if merge_grads:
            from ..channels.layout import _merge_grad_data
            data = _merge_grad_data(data)
            title = '%s RMS' % ch_type
        else:
            title = ch_type
        data = np.average(data, axis=1)
        axarr[0][idx].set_title(title)
        plot_topomap(data, pos, axes=axarr[0][idx], show=False)

    fig.suptitle('Average over %.2fs - %.2fs' % (xmin, xmax), fontsize=15,
                 y=0.1)
    tight_layout(pad=2.0, fig=fig)
    plt_show()
    if text is not None:
        text.set_visible(False)
        close_callback = partial(_topo_closed, ax=ax, lines=vert_lines,
                                 fill=fill)
        fig.canvas.mpl_connect('close_event', close_callback)
        evoked_fig.canvas.draw()
        evoked_fig.canvas.flush_events()


def _topo_closed(events, ax, lines, fill):
    """Callback for removing lines from evoked plot as topomap is closed."""
    for line in lines:
        ax.lines.remove(line[0])
    ax.collections.remove(fill)
    ax.get_figure().canvas.draw()


def _rgb(info, x, y, z):
    """Helper to transform x, y, z values into RGB colors"""
    all_pos = np.array([ch['loc'][:3] for ch in info['chs']])
    for idx, dim in enumerate([x, y, z]):
        this_pos = all_pos[:, idx]
        dim_min = this_pos.min()
        dim_max = (this_pos - dim_min).max()
        dim -= dim_min
        dim /= dim_max
    return np.asarray([x, y, z]).T


def _plot_legend(pos, colors, axis, bads, outlines):
    """Helper function to plot color/channel legends for butterfly plots
    with spatial colors"""
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    bbox = axis.get_window_extent()  # Determine the correct size.
    ratio = bbox.width / bbox.height
    ax = inset_axes(axis, width=str(30 / ratio) + '%', height='30%', loc=2)
    pos_x, pos_y = _prepare_topomap(pos, ax)
    ax.scatter(pos_x, pos_y, color=colors, s=25, marker='.', zorder=1)
    for idx in bads:
        ax.scatter(pos_x[idx], pos_y[idx], s=5, marker='.', color='w',
                   zorder=1)

    if isinstance(outlines, dict):
        _draw_outlines(ax, outlines)


def _plot_evoked(evoked, picks, exclude, unit, show,
                 ylim, proj, xlim, hline, units,
                 scalings, titles, axes, plot_type,
                 cmap=None, gfp=False, window_title=None,
                 spatial_colors=False, set_tight_layout=True,
                 selectable=True):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings)

    Extra param is:

    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.widgets import SpanSelector
    info = evoked.info
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')
    if isinstance(gfp, string_types) and gfp != 'only':
        raise ValueError('gfp must be boolean or "only". Got %s' % gfp)

    scalings = _handle_default('scalings', scalings)
    titles = _handle_default('titles', titles)
    units = _handle_default('units', units)
    # Valid data types ordered for consistency
    valid_channel_types = ['eeg', 'grad', 'mag', 'seeg', 'eog', 'ecg', 'emg',
                           'dipole', 'gof', 'bio', 'ecog']

    if picks is None:
        picks = list(range(info['nchan']))

    bad_ch_idx = [info['ch_names'].index(ch) for ch in info['bads']
                  if ch in info['ch_names']]
    if len(exclude) > 0:
        if isinstance(exclude, string_types) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list) and
              all(isinstance(ch, string_types) for ch in exclude)):
            exclude = [info['ch_names'].index(ch) for ch in exclude]
        else:
            raise ValueError('exclude has to be a list of channel names or '
                             '"bads"')

        picks = list(set(picks).difference(exclude))
    picks = np.array(picks)

    types = np.array([channel_type(info, idx) for idx in picks])
    n_channel_types = 0
    ch_types_used = []
    for t in valid_channel_types:
        if t in types:
            n_channel_types += 1
            ch_types_used.append(t)

    axes_init = axes  # remember if axes were given as input

    fig = None
    if axes is None:
        fig, axes = plt.subplots(n_channel_types, 1)

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if axes_init is not None:
        fig = axes[0].get_figure()
    if window_title is not None:
        fig.canvas.set_window_title(window_title)

    if not len(axes) == n_channel_types:
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%d: %s)' % (len(axes), n_channel_types,
                                             sorted(ch_types_used)))

    # instead of projecting during each iteration let's use the mixin here.
    if proj is True and evoked.proj is not True:
        evoked = evoked.copy()
        evoked.apply_proj()

    times = 1e3 * evoked.times  # time in milliseconds
    texts = list()
    idxs = list()
    lines = list()
    selectors = list()  # for keeping reference to span_selectors
    path_effects = [patheffects.withStroke(linewidth=2, foreground="w",
                                           alpha=0.75)]
    gfp_path_effects = [patheffects.withStroke(linewidth=5, foreground="w",
                                               alpha=0.75)]
    for ax, t in zip(axes, ch_types_used):
        line_list = list()  # 'line_list' contains the lines for this axes
        ch_unit = units[t]
        this_scaling = scalings[t]
        if unit is False:
            this_scaling = 1.0
            ch_unit = 'NA'  # no unit
        idx = list(picks[types == t])
        idxs.append(idx)
        if len(idx) > 0:
            # Set amplitude scaling
            D = this_scaling * evoked.data[idx, :]
            # Parameters for butterfly interactive plots
            if plot_type == 'butterfly':
                text = ax.annotate('Loading...', xy=(0.01, 0.1),
                                   xycoords='axes fraction', fontsize=20,
                                   color='green', zorder=3)
                text.set_visible(False)
                if selectable:
                    callback_onselect = partial(
                        _butterfly_onselect, ch_types=ch_types_used,
                        evoked=evoked, text=text)
                    blit = False if plt.get_backend() == 'MacOSX' else True
                    selectors.append(SpanSelector(
                        ax, callback_onselect, 'horizontal', minspan=10,
                        useblit=blit, rectprops=dict(alpha=0.5,
                                                     facecolor='red')))

                gfp_only = (isinstance(gfp, string_types) and gfp == 'only')
                if not gfp_only:
                    if spatial_colors:
                        chs = [info['chs'][i] for i in idx]
                        locs3d = np.array([ch['loc'][:3] for ch in chs])
                        x, y, z = locs3d.T
                        colors = _rgb(info, x, y, z)
                        if t in ('meg', 'mag', 'grad', 'eeg'):
                            layout = find_layout(info, ch_type=t, exclude=[])
                        else:
                            layout = find_layout(info, None, exclude=[])
                        # drop channels that are not in the data

                        used_nm = np.array(_clean_names(info['ch_names']))[idx]
                        names = np.asarray([name for name in used_nm
                                            if name in layout.names])
                        name_idx = [layout.names.index(name) for name in names]
                        if len(name_idx) < len(chs):
                            warn('Could not find layout for all the channels. '
                                 'Legend for spatial colors not drawn.')
                        else:
                            # find indices for bads
                            bads = [np.where(names == bad)[0][0] for bad in
                                    info['bads'] if bad in names]
                            pos, outlines = _check_outlines(layout.pos[:, :2],
                                                            'skirt', None)
                            pos = pos[name_idx]
                            _plot_legend(pos, colors, ax, bads, outlines)
                    else:
                        colors = ['k'] * len(idx)
                        for i in bad_ch_idx:
                            if i in idx:
                                colors[idx.index(i)] = 'r'
                    for ch_idx in range(len(D)):
                        line_list.append(ax.plot(times, D[ch_idx], picker=3.,
                                                 zorder=1,
                                                 color=colors[ch_idx])[0])
                if gfp:  # 'only' or boolean True
                    gfp_color = 3 * (0.,) if spatial_colors else (0., 1., 0.)
                    this_gfp = np.sqrt((D * D).mean(axis=0))
                    this_ylim = ax.get_ylim() if (ylim is None or t not in
                                                  ylim.keys()) else ylim[t]
                    if not gfp_only:
                        y_offset = this_ylim[0]
                    else:
                        y_offset = 0.
                    this_gfp += y_offset
                    ax.fill_between(times, y_offset, this_gfp, color='none',
                                    facecolor=gfp_color, zorder=1, alpha=0.25)
                    line_list.append(ax.plot(times, this_gfp, color=gfp_color,
                                             zorder=3)[0])
                    ax.text(times[0] + 0.01 * (times[-1] - times[0]),
                            this_gfp[0] + 0.05 * np.diff(ax.get_ylim())[0],
                            'GFP', zorder=4, color=gfp_color,
                            path_effects=gfp_path_effects)
                for ii, line in zip(idx, line_list):
                    if ii in bad_ch_idx:
                        line.set_zorder(2)
                        if spatial_colors:
                            line.set_linestyle("--")
                ax.set_ylabel('data (%s)' % ch_unit)
                # for old matplotlib, we actually need this to have a bounding
                # box (!), so we have to put some valid text here, change
                # alpha and path effects later
                texts.append(ax.text(0, 0, 'blank', zorder=3,
                                     verticalalignment='baseline',
                                     horizontalalignment='left',
                                     fontweight='bold', alpha=0))
            elif plot_type == 'image':
                im = ax.imshow(D, interpolation='nearest', origin='lower',
                               extent=[times[0], times[-1], 0, D.shape[0]],
                               aspect='auto', cmap=cmap)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.set_title(ch_unit)
                ax.set_ylabel('channels (%s)' % 'index')
            else:
                raise ValueError("plot_type has to be 'butterfly' or 'image'."
                                 "Got %s." % plot_type)
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and t in ylim:
                if plot_type == 'butterfly':
                    ax.set_ylim(ylim[t])
                elif plot_type == 'image':
                    im.set_clim(ylim[t])
            ax.set_title(titles[t] + ' (%d channel%s)' % (
                         len(D), 's' if len(D) > 1 else ''))
            ax.set_xlabel('time (ms)')

            if (plot_type == 'butterfly') and (hline is not None):
                for h in hline:
                    c = ('r' if not spatial_colors else 'grey')
                    ax.axhline(h, linestyle='--', linewidth=2, color=c)
        lines.append(line_list)
    if plot_type == 'butterfly':
        params = dict(axes=axes, texts=texts, lines=lines,
                      ch_names=info['ch_names'], idxs=idxs, need_draw=False,
                      path_effects=path_effects, selectors=selectors)
        fig.canvas.mpl_connect('pick_event',
                               partial(_butterfly_onpick, params=params))
        fig.canvas.mpl_connect('button_press_event',
                               partial(_butterfly_on_button_press,
                                       params=params))

    if axes_init is None:
        plt.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=info['projs'], axes=axes,
                      types=types, units=units, scalings=scalings, unit=unit,
                      ch_types_used=ch_types_used, picks=picks,
                      plot_update_proj_callback=_plot_update_evoked,
                      plot_type=plot_type)
        _draw_proj_checkbox(None, params)

    plt_show(show)
    fig.canvas.draw()  # for axes plots update axes.
    if set_tight_layout:
        tight_layout(fig=fig)

    return fig


def plot_evoked(evoked, picks=None, exclude='bads', unit=True, show=True,
                ylim=None, xlim='tight', proj=False, hline=None, units=None,
                scalings=None, titles=None, axes=None, gfp=False,
                window_title=None, spatial_colors=False):
    """Plot evoked data using butteryfly plots

    Left click to a line shows the channel name. Selecting an area by clicking
    and holding left mouse button plots a topographic map of the painted area.

    Note: If bad channels are not excluded they are shown in red.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : array-like of int | None
        The indices of channels to plot. If None show all.
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Show figure if True.
    ylim : dict | None
        ylim for plots (after scaling has been applied). e.g.
        ylim = dict(eeg=[-20, 20])
        Valid keys are eeg, mag, grad, misc. If None, the ylim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        xlim for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    hline : list of floats | None
        The values at which to show an horizontal line.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
    axes : instance of Axis | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    gfp : bool | 'only'
        Plot GFP in green if True or "only". If "only", then the individual
        channel traces will not be shown.
    window_title : str | None
        The title to put at the top of the figure.
    spatial_colors : bool
        If True, the lines are color coded by mapping physical sensor
        coordinates into color values. Spatially similar channels will have
        similar colors. Bad channels will be dotted. If False, the good
        channels are plotted black and bad channels red. Defaults to False.


    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure containing the butterfly plots.
    """
    return _plot_evoked(evoked=evoked, picks=picks, exclude=exclude, unit=unit,
                        show=show, ylim=ylim, proj=proj, xlim=xlim,
                        hline=hline, units=units, scalings=scalings,
                        titles=titles, axes=axes, plot_type="butterfly",
                        gfp=gfp, window_title=window_title,
                        spatial_colors=spatial_colors, selectable=True)


def plot_evoked_topo(evoked, layout=None, layout_scale=0.945, color=None,
                     border='none', ylim=None, scalings=None, title=None,
                     proj=False, vline=[0.0], fig_facecolor='k',
                     fig_background=None, axis_facecolor='k', font_color='w',
                     merge_grads=False, show=True):
    """Plot 2D topography of evoked responses.

    Clicking on the plot of an individual sensor opens a new figure showing
    the evoked response for the selected sensor.

    Parameters
    ----------
    evoked : list of Evoked | Evoked
        The evoked response to plot.
    layout : instance of Layout | None
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    layout_scale: float
        Scaling factor for adjusting the relative size of the layout
        on the canvas
    color : list of color objects | color object | None
        Everything matplotlib accepts to specify colors. If not list-like,
        the color specified will be repeated. If None, colors are
        automatically drawn.
    border : str
        matplotlib borders style to be used for each sensor plot.
    ylim : dict | None
        ylim for plots (after scaling has been applied). The value
        determines the upper and lower subplot limits. e.g.
        ylim = dict(eeg=[-20, 20]). Valid keys are eeg, mag, grad, misc.
        If None, the ylim parameter for each channel is determined by
        the maximum absolute peak.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    title : str
        Title of the figure.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    vline : list of floats | None
        The values at which to show a vertical line.
    fig_facecolor : str | obj
        The figure face color. Defaults to black.
    fig_background : None | numpy ndarray
        A background image for the figure. This must work with a call to
        plt.imshow. Defaults to None.
    axis_facecolor : str | obj
        The face color to be used for each sensor plot. Defaults to black.
    font_color : str | obj
        The color of text in the colorbar and title. Defaults to white.
    merge_grads : bool
        Whether to use RMS value of gradiometer pairs. Only works for Neuromag
        data. Defaults to False.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Images of evoked responses at sensor locations
    """
    return _plot_evoked_topo(evoked=evoked, layout=layout,
                             layout_scale=layout_scale, color=color,
                             border=border, ylim=ylim, scalings=scalings,
                             title=title, proj=proj, vline=vline,
                             fig_facecolor=fig_facecolor,
                             fig_background=fig_background,
                             axis_facecolor=axis_facecolor,
                             font_color=font_color, merge_grads=merge_grads,
                             show=show)


def _animate_evoked_topomap(evoked, ch_type='mag', times=None, frame_rate=None,
                            butterfly=False, blit=True, show=True):
    """Make animation of evoked data as topomap timeseries. Animation can be
    paused/resumed with left mouse button. Left and right arrow keys can be
    used to move backward or forward in time

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    ch_type : str | None
        Channel type to plot. Accepted data types: 'mag', 'grad', 'eeg'.
        If None, first available channel type from ('mag', 'grad', 'eeg') is
        used. Defaults to None.
    times : array of floats | None
        The time points to plot. If None, 10 evenly spaced samples are
        calculated over the evoked time series. Defaults to None.
    frame_rate : int | None
        Frame rate for the animation in Hz. If None, frame rate = sfreq / 10.
        Defaults to None.
    butterfly : bool
        Whether to plot the data as butterfly plot under the topomap.
        Defaults to False.
    blit : bool
        Whether to use blit to optimize drawing. In general, it is recommended
        to use blit in combination with ``show=True``. If you intend to save
        the animation it is better to disable blit. Defaults to True.
    show : bool
        Whether to show the animation. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib figure
        The figure.
    anim : instance of matplotlib FuncAnimation
        Animation of the topomap.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    return _topomap_animation(evoked, ch_type=ch_type, times=times,
                              frame_rate=frame_rate, butterfly=butterfly,
                              blit=blit, show=show)


def plot_evoked_image(evoked, picks=None, exclude='bads', unit=True, show=True,
                      clim=None, xlim='tight', proj=False, units=None,
                      scalings=None, titles=None, axes=None, cmap='RdBu_r'):
    """Plot evoked data as images

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    picks : array-like of int | None
        The indices of channels to plot. If None show all.
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Show figure if True.
    clim : dict | None
        clim for plots (after scaling has been applied). e.g.
        clim = dict(eeg=[-20, 20])
        Valid keys are eeg, mag, grad, misc. If None, the clim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        xlim for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    units : dict | None
        The units of the channel types used for axes lables. If None,
        defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
    axes : instance of Axis | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
    cmap : matplotlib colormap
        Colormap.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure containing the images.
    """
    return _plot_evoked(evoked=evoked, picks=picks, exclude=exclude, unit=unit,
                        show=show, ylim=clim, proj=proj, xlim=xlim,
                        hline=None, units=units, scalings=scalings,
                        titles=titles, axes=axes, plot_type="image",
                        cmap=cmap)


def _plot_update_evoked(params, bools):
    """ update the plot evoked lines
    """
    picks, evoked = [params[k] for k in ('picks', 'evoked')]
    times = evoked.times * 1e3
    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]
    params['proj_bools'] = bools
    new_evoked = evoked.copy()
    new_evoked.info['projs'] = []
    new_evoked.add_proj(projs)
    new_evoked.apply_proj()
    for ax, t in zip(params['axes'], params['ch_types_used']):
        this_scaling = params['scalings'][t]
        idx = [picks[i] for i in range(len(picks)) if params['types'][i] == t]
        D = this_scaling * new_evoked.data[idx, :]
        if params['plot_type'] == 'butterfly':
            for line, di in zip(ax.lines, D):
                line.set_data(times, di)
        else:
            ax.images[0].set_data(D)
    params['fig'].canvas.draw()


def plot_evoked_white(evoked, noise_cov, show=True):
    """Plot whitened evoked response

    Plots the whitened evoked response and the whitened GFP as described in
    [1]_. If one single covariance object is passed, the GFP panel (bottom)
    will depict different sensor types. If multiple covariance objects are
    passed as a list, the left column will display the whitened evoked
    responses for each channel based on the whitener from the noise covariance
    that has the highest log-likelihood. The left column will depict the
    whitened GFPs based on each estimator separately for each sensor type.
    Instead of numbers of channels the GFP display shows the estimated rank.
    Note. The rank estimation will be printed by the logger for each noise
    covariance estimator that is passed.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked response.
    noise_cov : list | instance of Covariance | str
        The noise covariance as computed by ``mne.cov.compute_covariance``.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.

    References
    ----------
    .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
           covariance estimation and spatial whitening of MEG and EEG
           signals, vol. 108, 328-342, NeuroImage.
    """
    return _plot_evoked_white(evoked=evoked, noise_cov=noise_cov,
                              scalings=None, rank=None, show=show)


def _plot_evoked_white(evoked, noise_cov, scalings=None, rank=None, show=True):
    """helper to plot_evoked_white

    Additional Parameters
    ---------------------
    scalings : dict | None
        The rescaling method to be applied to improve the accuracy of rank
        estimaiton. If dict, it will override the following default values
        (used if None)::

            dict(mag=1e12, grad=1e11, eeg=1e5)

        Note. Theses values were tested on different datests across various
        conditions. You should not need to update them.

    rank : dict of int | None
        Dict of ints where keys are 'eeg', 'mag' or 'grad'. If None,
        the rank is detected automatically. Defaults to None. Note.
        The rank estimation will be printed by the logger for each noise
        covariance estimator that is passed.

    """

    from ..cov import whiten_evoked, read_cov  # recursive import
    from ..cov import _estimate_rank_meeg_cov
    import matplotlib.pyplot as plt
    if scalings is None:
        scalings = dict(mag=1e12, grad=1e11, eeg=1e5)

    ch_used = [ch for ch in ['eeg', 'grad', 'mag'] if ch in evoked]
    has_meg = 'mag' in ch_used and 'grad' in ch_used

    if isinstance(noise_cov, string_types):
        noise_cov = read_cov(noise_cov)
    if not isinstance(noise_cov, (list, tuple)):
        noise_cov = [noise_cov]

    proc_history = evoked.info.get('proc_history', [])
    has_sss = False
    if len(proc_history) > 0:
        # if SSSed, mags and grad are not longer independent
        # for correct display of the whitening we will drop the cross-terms
        # (the gradiometer * magnetometer covariance)
        has_sss = 'max_info' in proc_history[0] and has_meg
    if has_sss:
        logger.info('SSS has been applied to data. Showing mag and grad '
                    'whitening jointly.')

    evoked = evoked.copy()  # handle ref meg
    evoked.info['projs'] = []  # either applied already or not-- else issue

    picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=False,
                       exclude='bads')
    evoked.pick_channels([evoked.ch_names[k] for k in picks])
    # important to re-pick. will otherwise crash on systems with ref channels
    # as first sensor block
    picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=False,
                       exclude='bads')

    picks_list = _picks_by_type(evoked.info, meg_combined=has_sss)
    if has_meg and has_sss:
        # reduce ch_used to combined mag grad
        ch_used = list(zip(*picks_list))[0]
    # order pick list by ch_used (required for compat with plot_evoked)
    picks_list = [x for x, y in sorted(zip(picks_list, ch_used))]
    n_ch_used = len(ch_used)

    # make sure we use the same rank estimates for GFP and whitening
    rank_list = []
    for cov in noise_cov:
        rank_ = {}
        C = cov['data'].copy()
        picks_list2 = [k for k in picks_list]
        if rank is None:
            if has_meg and not has_sss:
                picks_list2 += _picks_by_type(evoked.info,
                                              meg_combined=True)
            for ch_type, this_picks in picks_list2:
                this_info = pick_info(evoked.info, this_picks)
                idx = np.ix_(this_picks, this_picks)
                this_rank = _estimate_rank_meeg_cov(C[idx], this_info,
                                                    scalings)
                rank_[ch_type] = this_rank
        if rank is not None:
            rank_.update(rank)
        rank_list.append(rank_)
    evokeds_white = [whiten_evoked(evoked, n, picks, rank=r)
                     for n, r in zip(noise_cov, rank_list)]

    axes_evoked = None

    def whitened_gfp(x, rank=None):
        """Whitened Global Field Power

        The MNE inverse solver assumes zero mean whitened data as input.
        Therefore, a chi^2 statistic will be best to detect model violations.
        """
        return np.sum(x ** 2, axis=0) / (len(x) if rank is None else rank)

    # prepare plot
    if len(noise_cov) > 1:
        n_columns = 2
        n_extra_row = 0
    else:
        n_columns = 1
        n_extra_row = 1

    n_rows = n_ch_used + n_extra_row
    fig, axes = plt.subplots(n_rows,
                             n_columns, sharex=True, sharey=False,
                             figsize=(8.8, 2.2 * n_rows))
    if n_columns > 1:
        suptitle = ('Whitened evoked (left, best estimator = "%s")\n'
                    'and global field power '
                    '(right, comparison of estimators)' %
                    noise_cov[0].get('method', 'empirical'))
        fig.suptitle(suptitle)

    ax_gfp = None
    if any(((n_columns == 1 and n_ch_used == 1),
            (n_columns == 1 and n_ch_used > 1),
            (n_columns == 2 and n_ch_used == 1))):
        axes_evoked = axes[:n_ch_used]
        ax_gfp = axes[-1:]
    elif n_columns == 2 and n_ch_used > 1:
        axes_evoked = axes[:n_ch_used, 0]
        ax_gfp = axes[:, 1]
    else:
        raise RuntimeError('Wrong axes inputs')

    times = evoked.times * 1e3
    titles_ = _handle_default('titles')
    if has_sss:
        titles_['meg'] = 'MEG (combined)'

    colors = [plt.cm.Set1(i) for i in np.linspace(0, 0.5, len(noise_cov))]
    ch_colors = {'eeg': 'black', 'mag': 'blue', 'grad': 'cyan',
                 'meg': 'steelblue'}
    iter_gfp = zip(evokeds_white, noise_cov, rank_list, colors)

    if not has_sss:
        evokeds_white[0].plot(unit=False, axes=axes_evoked,
                              hline=[-1.96, 1.96], show=False)
    else:
        for ((ch_type, picks), ax) in zip(picks_list, axes_evoked):
            ax.plot(times, evokeds_white[0].data[picks].T, color='k')
            for hline in [-1.96, 1.96]:
                ax.axhline(hline, color='red', linestyle='--')

    # Now plot the GFP
    for evoked_white, noise_cov, rank_, color in iter_gfp:
        i = 0
        for ch, sub_picks in picks_list:
            this_rank = rank_[ch]
            title = '{0} ({2}{1})'.format(
                    titles_[ch] if n_columns > 1 else ch,
                    this_rank, 'rank ' if n_columns > 1 else '')
            label = noise_cov.get('method', 'empirical')

            ax_gfp[i].set_title(title if n_columns > 1 else
                                'whitened global field power (GFP),'
                                ' method = "%s"' % label)

            data = evoked_white.data[sub_picks]
            gfp = whitened_gfp(data, rank=this_rank)
            ax_gfp[i].plot(times, gfp,
                           label=(label if n_columns > 1 else title),
                           color=color if n_columns > 1 else ch_colors[ch])
            ax_gfp[i].set_xlabel('times [ms]')
            ax_gfp[i].set_ylabel('GFP [chi^2]')
            ax_gfp[i].set_xlim(times[0], times[-1])
            ax_gfp[i].set_ylim(0, 10)
            ax_gfp[i].axhline(1, color='red', linestyle='--')
            if n_columns > 1:
                i += 1

    ax = ax_gfp[0]
    if n_columns == 1:
        ax.legend(  # mpl < 1.2.1 compatibility: use prop instead of fontsize
            loc='upper right', bbox_to_anchor=(0.98, 0.9), prop=dict(size=12))
    else:
        ax.legend(loc='upper right', prop=dict(size=10))
        params = dict(top=[0.69, 0.82, 0.87][n_rows - 1],
                      bottom=[0.22, 0.13, 0.09][n_rows - 1])
        if has_sss:
            params['hspace'] = 0.49
        fig.subplots_adjust(**params)
    fig.canvas.draw()

    plt_show(show)
    return fig


def plot_snr_estimate(evoked, inv, show=True):
    """Plot a data SNR estimate

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance. This should probably be baseline-corrected.
    inv : instance of InverseOperator
        The minimum-norm inverse operator.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    from ..minimum_norm import estimate_snr
    snr, snr_est = estimate_snr(evoked, inv, verbose=True)
    fig, ax = plt.subplots(1, 1)
    lims = np.concatenate([evoked.times[[0, -1]], [-1, snr_est.max()]])
    ax.plot([0, 0], lims[2:], 'k:')
    ax.plot(lims[:2], [0, 0], 'k:')
    # Colors are "bluish green" and "vermilion" taken from:
    #  http://bconnelly.net/2013/10/creating-colorblind-friendly-figures/
    ax.plot(evoked.times, snr_est, color=[0.0, 0.6, 0.5])
    ax.plot(evoked.times, snr, color=[0.8, 0.4, 0.0])
    ax.set_xlim(lims[:2])
    ax.set_ylim(lims[2:])
    ax.set_ylabel('SNR')
    ax.set_xlabel('Time (sec)')
    if evoked.comment is not None:
        ax.set_title(evoked.comment)
    plt.draw()
    plt_show(show)
    return fig


def _connection_line(x, fig, sourceax, targetax):
    """Helper function to connect time series and topolots"""
    from matplotlib.lines import Line2D
    transFigure = fig.transFigure.inverted()
    tf = fig.transFigure

    (xt, yt) = transFigure.transform(targetax.transAxes.transform([.5, .25]))
    (xs, _) = transFigure.transform(sourceax.transData.transform([x, 0]))
    (_, ys) = transFigure.transform(sourceax.transAxes.transform([0, 1]))
    return Line2D((xt, xs), (yt, ys), transform=tf, color='grey',
                  linestyle='-', linewidth=1.5, alpha=.66, zorder=0)


def plot_evoked_joint(evoked, times="peaks", title='', picks=None,
                      exclude=None,
                      show=True, ts_args=None, topomap_args=None):
    """Plot evoked data as butterfly plot and add topomaps for selected
    time points.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance.
    times : float | array of floats | "auto" | "peaks".
        The time point(s) to plot. If "auto", 5 evenly spaced topographies
        between the first and last time instant will be shown. If "peaks",
        finds time points automatically by checking for 3 local maxima in
        Global Field Power. Defaults to "peaks".
    title : str | None
        The title. If `None`, suppress printing channel type. Defaults to ''.
    picks : array-like of int | None
        The indices of channels to plot. If None show all. Defaults to None.
    exclude : None | list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded. Defaults to None.
    show : bool
        Show figure if True. Defaults to True.
    ts_args : None | dict
        A dict of `kwargs` that are forwarded to `evoked.plot` to
        style the butterfly plot. `axes` and `show` are ignored.
        If `spatial_colors` is not in this dict, `spatial_colors=True`
        will be passed. Beyond that, if ``None``, no customizable arguments
        will be passed. Defaults to ``None``.
    topomap_args : None | dict
        A dict of `kwargs` that are forwarded to `evoked.plot_topomap`
        to style the topomaps. `axes` and `show` are ignored. If `times`
        is not in this dict, automatic peak detection is used. Beyond that,
        if ``None`, no customizable arguments will be passed.
        Defaults to ``None``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure | list
        The figure object containing the plot. If `evoked` has multiple
        channel types, a list of figures, one for each channel type, is
        returned.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    import matplotlib.pyplot as plt

    if ts_args is None:
        ts_args = dict()
    if topomap_args is None:
        topomap_args = dict()

    # channel selection
    # simply create a new evoked object(s) with the desired channel selection
    evoked = evoked.copy()

    if picks is not None:
        pick_names = [evoked.info['ch_names'][pick] for pick in picks]
    else:  # only pick channels that are plotted
        picks = _pick_data_channels(evoked.info, exclude=[])
        pick_names = [evoked.info['ch_names'][pick] for pick in picks]
    evoked.pick_channels(pick_names)

    if exclude == 'bads':
        exclude = [ch for ch in evoked.info['bads']
                   if ch in evoked.info['ch_names']]
    if exclude is not None:
        evoked.drop_channels(exclude)

    info = evoked.info
    data_types = ['eeg', 'grad', 'mag', 'seeg', 'ecog']
    ch_types = set(ch_type for ch_type in data_types if ch_type in evoked)

    # if multiple sensor types: one plot per channel type, recursive call
    if len(ch_types) > 1:
        figs = list()
        for t in ch_types:  # pick only the corresponding channel type
            ev_ = evoked.copy().pick_channels(
                [info['ch_names'][idx] for idx in range(info['nchan'])
                 if channel_type(info, idx) == t])
            if len(set([channel_type(ev_.info, idx)
                        for idx in range(ev_.info['nchan'])
                        if channel_type(ev_.info, idx) in data_types])) > 1:
                raise RuntimeError('Possibly infinite loop due to channel '
                                   'selection problem. This should never '
                                   'happen! Please check your channel types.')
            figs.append(
                plot_evoked_joint(
                    ev_, times=times, title=title, show=show, ts_args=ts_args,
                    exclude=list(), topomap_args=topomap_args))
        return figs

    fig = plt.figure(figsize=(8.0, 4.2))

    # set up time points to show topomaps for
    times = _process_times(evoked, times, few=True)

    # butterfly/time series plot
    # most of this code is about passing defaults on demand
    ts_ax = fig.add_subplot(212)
    ts_args_pass = dict((k, v) for k, v in ts_args.items() if k not in
                        ['axes', 'show', 'colorbar', 'set_tight_layout'])
    ts_args_def = dict(picks=None, unit=True, ylim=None, xlim='tight',
                       proj=False, hline=None, units=None, scalings=None,
                       titles=None, gfp=False, window_title=None,
                       spatial_colors=True)
    for key in ts_args_def:
        if key not in ts_args:
            ts_args_pass[key] = ts_args_def[key]
    _plot_evoked(evoked, axes=ts_ax, show=False, plot_type='butterfly',
                 exclude=[], set_tight_layout=False, **ts_args_pass)

    # handle title
    # we use a new axis for the title to handle scaling of plots
    old_title = ts_ax.get_title()
    ts_ax.set_title('')
    if title is not None:
        title_ax = plt.subplot(4, 3, 2)
        title = ', '.join([title, old_title]) if len(title) > 0 else old_title
        title_ax.text(.5, .5, title, transform=title_ax.transAxes,
                      horizontalalignment='center',
                      verticalalignment='center')
        title_ax.axis('off')

    # prepare axes for topomap
    # slightly convoluted due to colorbar placement and for vertical alignment
    ts = len(times) + 2
    map_ax = [plt.subplot(4, ts, x + 2 + ts) for x in range(ts - 2)]
    cbar_ax = plt.subplot(4, 3 * (ts + 1), 6 * (ts + 1))

    # topomap
    topomap_args_pass = dict((k, v) for k, v in topomap_args.items() if
                             k not in ['times', 'axes', 'show', 'colorbar'])
    topomap_args_pass['outlines'] = (topomap_args['outlines'] if 'outlines'
                                     in topomap_args else 'skirt')
    evoked.plot_topomap(times=times, axes=map_ax, show=False,
                        colorbar=False, **topomap_args_pass)

    if topomap_args.get('colorbar', True):
        from matplotlib import ticker
        cbar = plt.colorbar(map_ax[0].images[0], cax=cbar_ax)
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()

    plt.subplots_adjust(left=.1, right=.93, bottom=.14,
                        top=1. if title is not None else 1.2)

    # connection lines
    # draw the connection lines between time series and topoplots
    tstimes = [timepoint * 1e3 for timepoint in times]
    lines = [_connection_line(timepoint, fig, ts_ax, map_ax_)
             for timepoint, map_ax_ in zip(tstimes, map_ax)]
    for line in lines:
        fig.lines.append(line)

    # mark times in time series plot
    for timepoint in tstimes:
        ts_ax.axvline(timepoint, color='grey', linestyle='-',
                      linewidth=1.5, alpha=.66, zorder=0)

    # show and return it
    plt_show(show)
    return fig
