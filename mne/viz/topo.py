"""Functions to plot M/EEG data on topo (one axes per channel)
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import warnings
from itertools import cycle
from functools import partial

import numpy as np
from scipy import ndimage

# XXX : don't import pyplot here or you will break the doc

from ..io.pick import channel_type, pick_types
from ..fixes import normalize_colors
from ..utils import _clean_names

from .utils import _mutable_defaults, _check_delayed_ssp  # , COLORS
from .utils import _draw_proj_checkbox


def iter_topography(pos=None, info=None, on_pick=None, fig=None,
                    fig_facecolor='w', axis_facecolor='w', layout_scale=None,
                    axis_spinecolor='k', layout=None):
    """ Create iterator over channel positions

    This function returns a generator that unpacks into
    a series of matplotlib axis objects and data / channel
    indices, both corresponding to the sensor positions
    of the related layout passed or inferred from the channel info.
    `iter_topography`, hence, allows to conveniently realize custom
    topography plots.

    Parameters
    ----------
    pos : array
        Array of sensor positions (e.g., from
        mne.channels.layout.find_layout(info).pos).
        If None, will be inferred from layout.
    info : instance of mne.io.meas_info.Info
        The measurement info.
    layout : instance of mne.layout.Layout | None
        The layout to use. If None, layout will be inferred from info.
    on_pick : callable | None
        The callback function to be invoked on clicking one
        of the axes. Is supposed to instantiate the following
        API: `function(axis, channel_index)`
    fig : matplotlib.figure.Figure | None
        The figure object to be considered. If None, a new
        figure will be created.
    fig_facecolor : str | obj
        The figure face color. Defaults to white.
    axis_facecolor : str | obj
        The axis face color. Defaults to white.
    axis_spinecolor : str | obj
        The axis spine color. Defaults to black.
    layout_scale: float | None
        Scaling factor for adjusting the relative size of the layout
        on the canvas. If None, nothing will be scaled.

    Returns
    -------
    A generator that can be unpacked into

    ax : matplotlib.axis.Axis
        The current axis of the topo plot.
    ch_dx : int
        The related channel index.
    """
    import matplotlib.pyplot as plt

    if layout is None:
        from ..channels.layout import find_layout
        layout = find_layout(info)

    if pos is None:
        pos = layout.pos.copy()
        if layout_scale is not None:
            pos[:, :2] *= layout_scale

    if fig is None:
        fig = plt.figure()

    if on_pick is not None:
        callback = partial(_plot_topo_onpick, show_func=on_pick)
        fig.canvas.mpl_connect('button_press_event', callback)

    fig.set_facecolor(fig_facecolor)

    ch_names = _clean_names(info['ch_names'])
    ch_types = dict((ch_name, channel_type(info, j))
                    for j, ch_name in enumerate(ch_names))

    for idx, ch_name in enumerate(info["ch_names"]):
        if ch_name in layout.names:
            ax = plt.axes(pos[layout.names.index(ch_name)])
            ch_idx = ch_names.index(ch_name)
            vars(ax)['_mne_ch_name'] = ch_name
            vars(ax)['_mne_ch_idx'] = ch_idx
            vars(ax)['_mne_ch_type'] = ch_types[ch_name]
            vars(ax)['_mne_ax_face_color'] = str(axis_facecolor)   # why str?
            yield ax


def _scale_layout(layout, layout_scale, ybound=0, zero=0):
    """Helper function to set scaling factor so subplots don't overlap"""

    pos = layout.pos.copy()

    if layout_scale is 'wide':

        def _make_lines(pos):
            return pos

        def _check_intersect(a, b):  # check if subplot rectangles overlap
            dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            return ((dx >= 0) and (dy >= 0))

    elif layout_scale is 'tight':

        def _make_lines(pos):  # find coordinates of spines (x and y axis)
            return [((p[0], p[1] + p[3] * .5),
                    (p[0] + p[2], p[1] + p[3] * .5),
                    (p[0] + p[2] * -zero, p[1] + p[3] / 2 * (1 - ybound)),
                    (p[0] + p[2] * -zero, p[1] + p[3] * (1 - ybound / 2)))
                    for p in pos]

        def _check_intersect(a, b):  # check if spines overlap
            return sum([((a[0][0] < b[2][0] and a[1][0] > b[2][0] and
                        a[0][1] > b[2][1] and a[0][1] < b[3][1]) or
                        (b[0][0] < a[2][0] and b[1][0] > a[2][0] and
                        b[0][1] > a[2][1] and b[0][1] < a[3][1]))])

    if isinstance(layout_scale, str):
        from itertools import combinations
        layout_scale = 0.1
        pos[:, :2] *= layout_scale
        while any(_check_intersect(a, b) for a, b in
                  combinations(_make_lines(pos), 2)):
            layout_scale += 0.01
            pos = layout.pos.copy()
            pos[:, :2] *= layout_scale
            if layout_scale > 4:
                raise ValueError("Layout auto scaler not converging. "
                                 "Set a value for layout_scale manually, "
                                 "and/or ensure sensor coverage does "
                                 "not include overlapping sensors.")

    elif not isinstance(layout_scale, str):
        layout_scale = float(layout_scale)
        pos[:, :2] *= layout_scale

    return pos, layout_scale


def _plot_spines(ax, xlim, ylim, x_label, y_label, xticks, yticks,
                 linewidth, fontsize, spine_color, ch_name=None,
                 plot_unit_labels=True, plot_tick_labels=True,
                 plot_ticks=True, is_legend=None,
                 plot_type=None):
    """Helper function to plot spines and ticks for topo plots"""

    import matplotlib.lines as mpll

    if is_legend is True:
        ch_name = ''

    ax.patch.set_alpha(0)

    if ch_name is not None:
        ax.set_title(ch_name)
        if plot_type == 'evoked':
            ax.title.set_position([0.8, 0.75])
        else:
            ax.title.set_position([0.8, 0.85])

#    ax.spines['right'].set_color('none')
#    ax.spines['top'].set_color('none')
    ax.spines['right'].cla()
    ax.spines['top'].cla()

    for pos in ['left', 'bottom']:
        if plot_type == 'evoked':
            ax.spines[pos].set_position('zero')
        ax.spines[pos].set_smart_bounds(True)
        ax.spines[pos].set_color(spine_color)
        ax.spines[pos].set_linewidth(linewidth)

    if plot_type is not 'evoked':
        ax.spines['bottom'].set_position(('data', ylim[0]))

    if plot_type == 'evoked':
        ax.spines['left'].set_bounds(yticks[0], yticks[-1])
    else:
        ax.spines['left'].set_bounds(*ylim)

    if plot_unit_labels is True:
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        if plot_type != 'evoked':
            ax.yaxis.set_label_coords(-0.25, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.3)
        elif plot_type == 'evoked':
            ax.yaxis.set_label_coords(-0.25, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.15)
    elif plot_unit_labels is False:
        ax.set_ylabel('')
        ax.set_xlabel('')

    if plot_ticks is True:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for tick in ax.get_xaxis().get_major_ticks() + \
                ax.get_yaxis().get_major_ticks():
            tick.set_pad(2. * fontsize / 3.)
            tick.label1 = tick._get_text1()

    if plot_tick_labels is True:
        # ax.set_xticklabels([('' if i % 2 else str(t))
                            # for i, t in enumerate(xticks)])
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
    else:
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.set_xticklabels([])
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        ax.set_yticklabels([])

    ax.xaxis.set_tick_params(width=linewidth, length=3, color=spine_color)
    ax.yaxis.set_tick_params(width=linewidth, length=3, color=spine_color)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        item.set_color(spine_color)

    [line.set_marker(mpll.TICKDOWN) for line in ax.get_xticklines()]
    [line.set_marker(mpll.TICKLEFT) for line in ax.get_yticklines()]


def _plot_topo(info=None, times=None, show_func=None, layout=None,
               vmin=None, vmax=None, ylim=None, colorbar=None,
               border='none', axis_facecolor='k', fig_facecolor='k',
               cmap='RdBu_r', layout_scale=None, title=None, x_label=None,
               y_label=None, vline=None, xticks=2, yticks=None,
               font_color=None, linewidth=1, internal_legend=False,
               external_legend=False, fontsize=None,
               ylim_dict=None, show_names=None, plot_type=None,
               external_scale=1):
    """Helper function to plot on sensor layout"""
    import matplotlib.pyplot as plt

    # prepare callbacks
    tmin, tmax = times[[0, -1]]
    on_pick = partial(show_func, tmin=tmin, tmax=tmax, vmin=vmin,
                      vmax=vmax, ylim=ylim, colorbar=colorbar)

    if xticks == 2:
        xticks = [0.0, (tmin if np.abs(tmin) > np.abs(tmax) else tmax) / 2]
        xticks = [round(.05 * round(float(x) / .05), 2) for x in xticks]
    elif isinstance(xticks, int):
        endpoints = times[0], times[-1]
        xticks = [round(.05 * round(float(x) / .05), 2)
                  for x in np.linspace(*endpoints, num=xticks + 2)][1:-1]
        if 0.0 in xticks and plot_type == 'evoked' and len(xticks) > 1:
            del xticks[xticks.index(0.0)]
#        if plot_type is None:
#            xticks = list(set([float(x) for x in xticks]))
        if isinstance(xticks, (float, int)):
            xticks = list(xticks)

    if isinstance(ylim_dict, (list, tuple)):
        from collections import defaultdict
        ys = [y for y in ylim_dict]
        ylim_dict = defaultdict(lambda: ys)

    if yticks is None:  # it would be better to put rounding 1 level up
        endpoints = (np.min(ylim_dict["dummy"]), np.max(ylim_dict["dummy"]))
        yticks = [round(.05 * round(float(x) / .05), 2)
                  for x in np.linspace(*endpoints, num=5)][1:-1]
        if plot_type is None:
            yticks = list(set([int(x) for x in yticks]))
        if 0.0 in yticks:  # and (plot_type == 'evoked' or endpoints[0] == 0):
            del yticks[yticks.index(0.0)]
    if any(np.abs(x) > 20 for x in yticks):
        yticks = [round(5 * round(float(x) / 5), 2)
                  for x in np.linspace(*endpoints, num=len(yticks))]
        yticks = [(y + 5 if y < endpoints[0] else y) for y in yticks]
        yticks = [(y - 5 if y > endpoints[1] else y) for y in yticks]

    from matplotlib.colors import colorConverter
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*colorConverter.to_rgb(fig_facecolor))
    spine_color = ('k' if l >= .5 else 'w')

    if font_color is None:
        font_color = spine_color

    fig = plt.figure()

    if (len(info["ch_names"]) < 35) and (show_names is None):
        show_names = True

    if layout_scale is None or layout_scale is 'auto':
        if len(info["ch_names"]) < 35:
            layout_scale = 'wide'
        elif len(info["ch_names"]) >= 35:
            layout_scale = 'tight'

    if plot_type is 'evoked':
        zero = tmin / (tmax - tmin)
        ybound = yticks[0] / ylim_dict["dummy"][0]
    else:
        zero = 0
        ybound = 0

    pos, layout_scale = _scale_layout(layout, layout_scale, ybound, zero)
    if fontsize is None:
        fontsize = 9 * layout_scale

    my_topo_plot = iter_topography(pos=pos, info=info, on_pick=on_pick,
                                   layout=layout,
                                   fig=fig, axis_spinecolor=border,
                                   axis_facecolor=axis_facecolor,
                                   fig_facecolor=fig_facecolor)

    for ax in my_topo_plot:

        if layout.kind == 'Vectorview-all' and ylim is not None:
            this_type = {'mag': 0, 'grad': 1}[channel_type(info,
                                              ax._mne_ch_idx)]
            ylim_ = [v[this_type] if _check_vlim(v) else v for v in ylim]
        else:
            ylim_ = ylim

        show_func(ax, ax._mne_ch_idx, tmin=tmin, tmax=tmax, vmin=vmin,
                  vmax=vmax, ylim=ylim_)

        if (axis_facecolor is fig_facecolor) or (axis_facecolor is None):
            ax.patch.set_alpha(0)
        else:
            ax.patch.set_facecolor(axis_facecolor)

        _plot_spines(ax, (tmin, tmax), ylim_dict[ax._mne_ch_type],
                     x_label, y_label, xticks, yticks, linewidth, fontsize,
                     spine_color, plot_type=plot_type,
                     plot_tick_labels=(True if internal_legend is True
                     else False),
                     plot_unit_labels=(True if internal_legend is True
                     and external_legend is False else False),
                     plot_ticks=(True if (internal_legend or external_legend)
                     else False),
                     ch_name=(ax._mne_ch_name
                              if show_names is True else None))

        if ylim_ and not any(v is None for v in ylim_):
            plt.ylim(*ylim_)

    if title is not None:
        plt.figtext(0.01, 1 * layout_scale, title, color=font_color,
                    fontsize=fontsize * 2)

    if colorbar:
        norm = normalize_colors(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax)[1:-1:2])
        ax = plt.axes((0.875 * layout_scale, 0.75 * layout_scale,
                      0.1 * layout_scale, 0.25 * layout_scale))
        cb = fig.colorbar(sm, ax=ax)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        plt.setp(cb_yticks, color=spine_color, fontsize=fontsize)
        cb.ax.get_children()[4].set_linewidths(0.4)
        ax.axis('off')

    if external_legend is True:
        legend_ax = plt.axes((0.025, 0.025, pos[0][2], pos[0][3]))

        # if plot_type == 'evoked':
        legend_ax.plot((tmin, tmax), ylim_dict["dummy"])
        legend_ax.lines.pop()
        # else:
        #    legend_ax.imshow(np.zeros((2,2)), extent=[tmin, tmax, 0,
        #                     ylim_dict["dummy"]], alpha=0, aspect='auto',
        #                     origin='lower')

        _plot_spines(legend_ax, (tmin, tmax), ylim_dict["dummy"],
                     x_label, y_label, xticks, yticks, linewidth,
                     fontsize, spine_color, plot_type=plot_type,
                     is_legend=True)

        return fig, legend_ax, layout_scale

    else:
        return fig, None, layout_scale


def _plot_topo_onpick(event, show_func=None, colorbar=False):
    """Onpick callback that shows a single channel in a new figure"""

    # make sure that the swipe gesture in OS-X doesn't open many figures
    orig_ax = event.inaxes
    if event.inaxes is None:
        return

    import matplotlib.pyplot as plt
    try:
        ch_idx = orig_ax._mne_ch_idx
        face_color = orig_ax._mne_ax_face_color
        fig, ax = plt.subplots(1)

        plt.title(orig_ax._mne_ch_name)
        ax.set_axis_bgcolor(face_color)

        # allow custom function to override parameters
        show_func(plt, ch_idx)

    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers,
        # so we print
        # it here to know what went wrong
        print(err)
        raise err


def _imshow_tfr(ax, ch_idx, tmin, tmax, vmin, vmax, ylim=None, tfr=None,
                freq=None, vline=None, x_label=None, y_label=None,
                colorbar=False, picker=True, cmap=None, title=None):
    """ Aux function to show time-freq map on topo """
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.jet

    extent = (tmin, tmax, freq[0], freq[-1])
    ax.imshow(tfr[ch_idx], extent=extent, aspect="auto", origin="lower",
              vmin=vmin, vmax=vmax, picker=picker, cmap=cmap)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if colorbar:
        plt.colorbar()
    if title:
        plt.title(title)


def _plot_timeseries(ax, ch_idx, tmin, tmax, vmin, vmax, ylim, data, color,
                     times, vline=None, linewidth=0.5, colorbar=False):
    """ Aux function to show time series on topo """
    import matplotlib.pyplot as plt
    picker_flag = False
    for ii, (data_, color_) in enumerate(zip(data, color)):
        if not picker_flag:
            # use large tol for picker so we can click anywhere in the axes
            ax.plot(times, data_[ch_idx], color_, picker=1e9,
                    linewidth=linewidth, zorder=20 + ii)
            picker_flag = True
        else:
            ax.plot(times, data_[ch_idx], color_, linewidth=linewidth,
                    zorder=20 + ii)
    if vline:
        cs = cycle(['pink', 'purple', 'green'])
        [plt.axvline(x, color=c,
         linewidth=linewidth) for x, c in zip(vline, cs)]
    if colorbar:
        plt.colorbar()  # will this ever be true?


def _check_vlim(vlim):
    """AUX function"""
    return not np.isscalar(vlim) and vlim is not None


def plot_topo(evoked, layout=None, layout_scale='auto', color=None,
              border='none', ylim=None, scalings=None, units=None, title=None,
              proj=False, vline=None, fig_facecolor='w', axis_facecolor=None,
              font_color=None, x_label='Time (s)', show_names=None,
              legend='external', xticks=1, yticks=None,
              linewidth=1, fontsize=9, conditions=None):
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
        automatically drawn. If not None, should be at least as long as
        `evoked`.
    border : str
        matplotlib borders style to be used for each sensor plot.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,
        defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    ylim : dict | None
        ylim for plots. The value determines the upper and lower subplot
        limits. e.g. ylim = dict(eeg=[-200e-6, 200e6]). Valid keys are eeg,
        mag, grad, misc. If None, the ylim parameter for all channels is
        determined by the maximum absolute peak. If an external legend is
        plotted, must be None or length 1.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    title : str
        Title of the figure.
    vline : list of floats | None
        The values at which to show a vertical line. Defaults to None.
    units : dict | str | None
        The units of the channel types used for axes lables, either as a single
        string or as dict by channel type. If None, defaults to
        `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    legend : None | str
        If not None: plot an external x and y axis legend (usually time/unit),
        showing and labelling time and unit ticks. Requires `evokeds` to hold
        only 1 channel type.
        If 'internal': plot x and y axis ticks and labels for each subplot.
        If 'external': plot an external legend.
        If 'both': plot an external legend, and plot tick marks for subplots.
    show_names : bool
        Should channel names be plotted next to each topo plot?
    x_label : string | None
        Label for x axis. Defaults to 'Time (s)'.
    fig_facecolor : str | obj
        The figure face color. Defaults to black. If "black", spines are
        white (else they are black).
    axis_facecolor : str | obj
        The face color to be used for each sensor plot. If none, transparent.
    font_color : str | obj
        The color of text in the colorbar and title. Defaults to spine_color
        (which is automatically set based on fig_facecolor if None).
    xticks : list of floats | int
        If list, list of tick marks for time axis. If int, number of ticks.
        If None, determined automatically.
    yticks : list of floats | None
        List of tick marks for y axis. If None, determined automatically.
    linewidth : float | None
        Linewidth for time series, spines, tick marks.
    conditions : list of str | None
        Condition labels. Will be plotted in the specified colors. Must be of
        the same length as `evokeds`.
    fontsize : float | None
        Font size for axis labels.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        Images of evoked responses at sensor locations
    l_ax : The axis instance holding the legend.
    """

    if not type(evoked) in (tuple, list):
        evoked = [evoked]

    if color is None:
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                 '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
        color = color[:len(evoked)]
    if 1 < len(color) < len(evoked):
        warnings.warn('More evoked objects than colors available.'
                      'You should pass a list of unique colors.')
        color = cycle(color)
    if color is 'bw':  # currently not implemented
        fig_facecolor = 'w'
        from itertools import cycle
        color = cycle('k')
        # line_styles = ['-', ':', '-.', '--']

    times = evoked[0].times

    if legend in ['internal', 'both']:
        internal_legend = True
    else:
        internal_legend = False
    if legend in ['external', 'both']:
        external_legend = True
    else:
        external_legend = False

    if not all([(e.times == times).all() for e in evoked]):
        raise ValueError('All evoked.times must be the same')

    info = evoked[0].info
    ch_names = evoked[0].ch_names
    if show_names is None and len(ch_names) > 35:
        show_names = False

    if not all([e.ch_names == ch_names for e in evoked]):
        raise ValueError('All evoked.picks must be the same')
    ch_names = _clean_names(ch_names)

    if layout is None:
        from ..channels.layout import find_layout
        layout = find_layout(info)

    # XXX. at the moment we are committed to 1- / 2-sensor-types layouts
    chs_in_layout = set(layout.names) & set(ch_names)
    types_used = set(channel_type(info, ch_names.index(ch))
                     for ch in chs_in_layout)
    # remove possible reference meg channels
    types_used = set.difference(types_used, set('ref_meg'))
    # one check for all vendors
    meg_types = set(('mag', 'grad'))
    is_meg = len(set.intersection(types_used, meg_types)) > 0
    if is_meg:
        types_used = list(types_used)[::-1]  # -> restore kwarg order
        picks = [pick_types(info, meg=kk, ref_meg=False, exclude=[])
                 for kk in types_used]
    else:
        types_used_kwargs = dict((t, True) for t in types_used)
        picks = [pick_types(info, meg=False, exclude=[], **types_used_kwargs)]
    assert isinstance(picks, list) and len(types_used) == len(picks)

    scalings = _mutable_defaults(('scalings', scalings))[0]
    if units is None:
        units = _mutable_defaults(('units', units))[0]
        if (external_legend is True or internal_legend is True):
            try:
                unit = units[list(types_used)[0]]
            except TypeError:
                raise TypeError('Legend plotting only with 1 channel type')
    elif isinstance(units, str):
        unit = units
    elif not isinstance(units, dict):
        raise TypeError('units must be a string, a dict or None')

    evoked = [e.copy() for e in evoked]
    for e in evoked:
        for pick, t in zip(picks, types_used):
            e.data[pick] = e.data[pick] * scalings[t]

    if proj is True and all([e.proj is not True for e in evoked]):
        evoked = [e.apply_proj() for e in evoked]
    elif proj == 'interactive':  # let it fail early.
        for e in evoked:
            _check_delayed_ssp(e)

    # sorry, y axis handling is a terrible mess right now
    from collections import defaultdict
    if ylim is None:
        def set_ylim(x):
            return np.abs(x).max()
        ylim_ = [set_ylim([e.data[t] for e in evoked]) for t in picks]
        ymax = np.max(np.array(ylim_))
        ylim_ = (-ymax, ymax)
        ylim_dict = defaultdict(lambda: ylim_)
    elif isinstance(ylim, (list, tuple)):
        ylim_ = (-max(ylim), max(ylim))
        ylim_dict = defaultdict(lambda: ylim_)
    elif isinstance(ylim, dict):
        if external_legend is True and len(ylim) > 1:
            ymax = np.max([np.abs(x) for i in ylim.values() for x in i])
            ylim_dict = defaultdict(lambda: (ymax, -ymax))
            warnings.warn('If ylim is a dict of len > 1, all channel types'
                          'will be plotted to the same (maximal) ylims.')
        else:
            ylim_dict = defaultdict(lambda: [v for v in ylim.values()][0])
        ylim_ = _mutable_defaults(('ylim', ylim))[0]
        ylim_ = [ylim_[kk] for kk in types_used]
        # extra unpack to avoid bug #1700
        if len(ylim_) == 1:
            ylim_ = ylim_[0]
        else:
            ylim_ = zip(*[np.array(yl) for yl in ylim_])
    else:
        raise ValueError('ylim must be None or a dict')

    plot_fun = partial(_plot_timeseries, data=[e.data for e in evoked],
                       color=color, times=times, vline=vline,
                       linewidth=linewidth)

    fig, l_ax, l_scale = _plot_topo(info=info, times=times, show_func=plot_fun,
                                    layout=layout, colorbar=False,
                                    ylim=ylim_, cmap=None,
                                    layout_scale=layout_scale,
                                    border=border, fig_facecolor=fig_facecolor,
                                    font_color=font_color,
                                    axis_facecolor=axis_facecolor,
                                    fontsize=fontsize,
                                    external_legend=(True if external_legend
                                                     is not False else False),
                                    internal_legend=internal_legend,
                                    plot_type='evoked', title=title,
                                    vline=vline, x_label=x_label,
                                    y_label=(unit if
                                             (external_legend or
                                              internal_legend)
                                             else None),
                                    show_names=show_names,
                                    xticks=xticks, yticks=yticks,
                                    ylim_dict=ylim_dict,
                                    external_scale=external_legend)

    if proj == 'interactive':
        for e in evoked:
            _check_delayed_ssp(e)
        params = dict(evokeds=evoked, times=times,
                      plot_update_proj_callback=_plot_update_evoked_topo,
                      projs=evoked[0].info['projs'], fig=fig)
        _draw_proj_checkbox(None, params)

    if conditions is not None:
        if len(conditions) != len(evoked):
            raise ValueError("Condition and Evokeds must have the same length")
        import matplotlib.pyplot as plt
        for cond, col, pos in zip(reversed(conditions), reversed(color),
                                  np.arange(0.1, 0.4, 0.025)):
            plt.figtext(l_scale, pos, cond, color=col,
                        fontsize=fontsize * l_scale * 1.5)

    return fig, l_ax


def _plot_update_evoked_topo(params, bools):
    """Helper function to update topo sensor plots"""
    evokeds, times, fig = [params[k] for k in ('evokeds', 'times', 'fig')]

    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]

    params['proj_bools'] = bools
    evokeds = [e.copy() for e in evokeds]
    for e in evokeds:
        e.info['projs'] = []
        e.add_proj(projs)
        e.apply_proj()

    # make sure to only modify the time courses, not the ticks
    axes = fig.get_axes()
    n_lines = len(axes[0].lines)
    n_diff = len(evokeds) - n_lines
    ax_slice = slice(abs(n_diff)) if n_diff < 0 else slice(n_lines)
    for ax in axes:
        lines = ax.lines[ax_slice]
        for line, evoked in zip(lines, evokeds):
            line.set_data(times, evoked.data[ax._mne_ch_idx])

    fig.canvas.draw()


def _erfimage_imshow(ax, ch_idx, tmin, tmax, vmin, vmax, ylim=None,
                     data=None, epochs=None, sigma=None,
                     order=None, scalings=None, vline=None,
                     x_label=None, y_label=None, colorbar=False,
                     cmap='RdBu_r'):
    """Aux function to plot erfimage on sensor topography"""

    import matplotlib.pyplot as plt
    this_data = data[:, ch_idx, :].copy()
    ch_type = channel_type(epochs.info, ch_idx)
    if ch_type not in scalings:
        raise KeyError('%s channel type not in scalings' % ch_type)
    this_data *= scalings[ch_type]

    if callable(order):
        order = order(epochs.times, this_data)

    if order is not None:
        this_data = this_data[order]

    this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

    ax.imshow(this_data, extent=[tmin, tmax, 0, len(data)], aspect='auto',
              origin='lower', vmin=vmin, vmax=vmax, picker=True,
              cmap=cmap)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if colorbar:
        plt.colorbar()


def plot_topo_image_epochs(epochs, layout=None, sigma=0.3, vmin=None,
                           vmax=None, colorbar=True, order=None, cmap='RdBu_r',
                           layout_scale='tight', title=None, scalings=None,
                           border='none', fig_facecolor='k', font_color=None,
                           y_label='Epoch', show_names=None, xticks=2,
                           legend='both', linewidth=1, fontsize=9):
    """Plot Event Related Potential / Fields image on topographies

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    layout: instance of Layout
        System specific sensor positions.
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image.
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
        (data.shape[1] == len(times)).
    cmap : instance of matplotlib.pyplot.colormap
        Colors to be mapped to the values.
    layout_scale: float
        scaling factor for adjusting the relative size of the layout
        on the canvas.
    title : str
        Title of the figure.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If
        None, defaults to `dict(eeg=1e6, grad=1e13, mag=1e15)`.
    border : str
        matplotlib borders style to be used for each sensor plot.
    fig_facecolor : str | obj
        The figure face color. Defaults to black. If "black", spines are
        white (else they are black).
    font_color : str | obj
        The color of text in the colorbar and title. Defaults to spine_color
        (which is automatically set based on fig_facecolor if None).
    y_label : 'str'
        Y axis label (also: sorting factor).
    show_names : bool
        Should channel names be plotted next to each topo plot?
    legend : None | str
        If not None: plot an external x and y axis legend (usually time/unit),
        showing and labelling time and unit ticks. Requires `evokeds` to hold
        only 1 channel type.
        If 'internal': plot x and y axis ticks and labels for each subplot.
        If 'external': plot an external legend.
        If 'both': plot an external legend, and plot tick marks for subplots.


    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    scalings = _mutable_defaults(('scalings', scalings))[0]
    data = epochs.get_data()
    if vmin is None:
        vmin = -np.max(np.abs(data))
    if vmax is None:
        vmax = np.max(np.abs(data))
    if layout is None:
        from ..channels.layout import find_layout
        layout = find_layout(epochs.info)

    erf_imshow = partial(_erfimage_imshow, scalings=scalings, order=order,
                         data=data, epochs=epochs, sigma=sigma,
                         cmap=cmap)

    fig, l_x, l_s = _plot_topo(info=epochs.info, times=epochs.times,
                               show_func=erf_imshow, layout=layout,
                               colorbar=colorbar, vmin=vmin, vmax=vmax,
                               cmap=cmap, layout_scale=layout_scale,
                               title=title, fig_facecolor=fig_facecolor,
                               ylim_dict=(0, epochs.get_data().shape[0]),
                               font_color=font_color, border=border,
                               x_label='Time (s)', y_label=y_label,
                               internal_legend = (True if legend in
                                                  ['internal', 'both']
                                                  else False),
                               external_legend = (True if legend in
                                                  ['external', 'both']
                                                  else False),
                               show_names=show_names,
                               linewidth=linewidth, xticks=xticks,
                               fontsize=fontsize)

    return fig
