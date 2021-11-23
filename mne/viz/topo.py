"""Functions to plot M/EEG data on topo (one axes per channel)."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
from functools import partial
from itertools import cycle

import numpy as np

from ..io.pick import channel_type, pick_types
from ..utils import _clean_names, warn, _check_option, Bunch, fill_doc, _to_rgb
from ..channels.layout import _merge_ch_data, _pair_grad_sensors, find_layout
from ..defaults import _handle_default
from .utils import (_check_delayed_ssp, _get_color_list, _draw_proj_checkbox,
                    add_background_image, plt_show, _setup_vmin_vmax,
                    DraggableColorbar, _setup_ax_spines,
                    _check_cov, _plot_masked_image)


@fill_doc
def iter_topography(info, layout=None, on_pick=None, fig=None,
                    fig_facecolor='k', axis_facecolor='k',
                    axis_spinecolor='k', layout_scale=None, legend=False):
    """Create iterator over channel positions.

    This function returns a generator that unpacks into
    a series of matplotlib axis objects and data / channel
    indices, both corresponding to the sensor positions
    of the related layout passed or inferred from the channel info.
    Hence, this enables convenient topography plot customization.

    Parameters
    ----------
    %(info_not_none)s
    layout : instance of mne.channels.Layout | None
        The layout to use. If None, layout will be guessed.
    on_pick : callable | None
        The callback function to be invoked on clicking one
        of the axes. Is supposed to instantiate the following
        API: ``function(axis, channel_index)``.
    fig : matplotlib.figure.Figure | None
        The figure object to be considered. If None, a new
        figure will be created.
    fig_facecolor : color
        The figure face color. Defaults to black.
    axis_facecolor : color
        The axis face color. Defaults to black.
    axis_spinecolor : color
        The axis spine color. Defaults to black. In other words,
        the color of the axis' edge lines.
    layout_scale : float | None
        Scaling factor for adjusting the relative size of the layout
        on the canvas. If None, nothing will be scaled.
    legend : bool
        If True, an additional axis is created in the bottom right corner
        that can be used to, e.g., construct a legend. The index of this
        axis will be -1.

    Returns
    -------
    gen : generator
        A generator that can be unpacked into:

        ax : matplotlib.axis.Axis
            The current axis of the topo plot.
        ch_dx : int
            The related channel index.
    """
    return _iter_topography(info, layout, on_pick, fig, fig_facecolor,
                            axis_facecolor, axis_spinecolor, layout_scale,
                            legend=legend)


def _legend_axis(pos):
    """Add a legend axis to the bottom right."""
    import matplotlib.pyplot as plt
    left, bottom = pos[:, 0].max(), pos[:, 1].min()
    wid, hei = pos[-1, 2:]
    return plt.axes([left, bottom + .05, wid, hei])


def _iter_topography(info, layout, on_pick, fig, fig_facecolor='k',
                     axis_facecolor='k', axis_spinecolor='k',
                     layout_scale=None, unified=False, img=False, axes=None,
                     legend=False):
    """Iterate over topography.

    Has the same parameters as iter_topography, plus:

    unified : bool
        If False (default), multiple matplotlib axes will be used.
        If True, a single axis will be constructed. The former is
        useful for custom plotting, the latter for speed.
    """
    from matplotlib import pyplot as plt, collections

    if fig is None:
        fig = plt.figure()

    def format_coord_unified(x, y, pos=None, ch_names=None):
        """Update status bar with channel name under cursor."""
        # find candidate channels (ones that are down and left from cursor)
        pdist = np.array([x, y]) - pos[:, :2]
        pind = np.where((pdist >= 0).all(axis=1))[0]
        if len(pind) > 0:
            # find the closest channel
            closest = pind[np.sum(pdist[pind, :]**2, axis=1).argmin()]
            # check whether we are inside its box
            in_box = (pdist[closest, :] < pos[closest, 2:]).all()
        else:
            in_box = False
        return (('%s (click to magnify)' % ch_names[closest]) if
                in_box else 'No channel here')

    def format_coord_multiaxis(x, y, ch_name=None):
        """Update status bar with channel name under cursor."""
        return '%s (click to magnify)' % ch_name

    fig.set_facecolor(fig_facecolor)
    if layout is None:
        layout = find_layout(info)

    if on_pick is not None:
        callback = partial(_plot_topo_onpick, show_func=on_pick)
        fig.canvas.mpl_connect('button_press_event', callback)

    pos = layout.pos.copy()
    if layout_scale:
        pos[:, :2] *= layout_scale

    ch_names = _clean_names(info['ch_names'])
    iter_ch = [(x, y) for x, y in enumerate(layout.names) if y in ch_names]
    if unified:
        if axes is None:
            under_ax = plt.axes([0, 0, 1, 1])
            under_ax.axis('off')
        else:
            under_ax = axes
        under_ax.format_coord = partial(format_coord_unified, pos=pos,
                                        ch_names=layout.names)
        under_ax.set(xlim=[0, 1], ylim=[0, 1])

        axs = list()
    for idx, name in iter_ch:
        ch_idx = ch_names.index(name)
        if not unified:  # old, slow way
            ax = plt.axes(pos[idx])
            ax.patch.set_facecolor(axis_facecolor)
            for spine in ax.spines.values():
                spine.set_color(axis_spinecolor)
            if not legend:
                ax.set(xticklabels=[], yticklabels=[])
                for tick in ax.get_xticklines() + ax.get_yticklines():
                    tick.set_visible(False)
            ax._mne_ch_name = name
            ax._mne_ch_idx = ch_idx
            ax._mne_ax_face_color = axis_facecolor
            ax.format_coord = partial(format_coord_multiaxis, ch_name=name)
            yield ax, ch_idx
        else:
            ax = Bunch(ax=under_ax, pos=pos[idx], data_lines=list(),
                       _mne_ch_name=name, _mne_ch_idx=ch_idx,
                       _mne_ax_face_color=axis_facecolor)
            axs.append(ax)
    if not unified and legend:
        ax = _legend_axis(pos)
        yield ax, -1

    if unified:
        under_ax._mne_axs = axs
        # Create a PolyCollection for the axis backgrounds
        verts = np.transpose([pos[:, :2],
                              pos[:, :2] + pos[:, 2:] * [1, 0],
                              pos[:, :2] + pos[:, 2:],
                              pos[:, :2] + pos[:, 2:] * [0, 1],
                              ], [1, 0, 2])
        if not img:
            under_ax.add_collection(collections.PolyCollection(
                verts, facecolor=axis_facecolor, edgecolor=axis_spinecolor,
                linewidth=1.))  # Not needed for image plots.
        for ax in axs:
            yield ax, ax._mne_ch_idx


def _plot_topo(info, times, show_func, click_func=None, layout=None,
               vmin=None, vmax=None, ylim=None, colorbar=None, border='none',
               axis_facecolor='k', fig_facecolor='k', cmap='RdBu_r',
               layout_scale=None, title=None, x_label=None, y_label=None,
               font_color='w', unified=False, img=False, axes=None):
    """Plot on sensor layout."""
    import matplotlib.pyplot as plt

    if layout.kind == 'custom':
        layout = deepcopy(layout)
        layout.pos[:, :2] -= layout.pos[:, :2].min(0)
        layout.pos[:, :2] /= layout.pos[:, :2].max(0)

    # prepare callbacks
    tmin, tmax = times[0], times[-1]
    click_func = show_func if click_func is None else click_func
    on_pick = partial(click_func, tmin=tmin, tmax=tmax, vmin=vmin,
                      vmax=vmax, ylim=ylim, x_label=x_label,
                      y_label=y_label)

    if axes is None:
        fig = plt.figure()
        axes = plt.axes([0.015, 0.025, 0.97, 0.95])
        axes.set_facecolor(fig_facecolor)
    else:
        fig = axes.figure
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm.set_array(np.linspace(vmin, vmax))
        cb = fig.colorbar(sm, ax=axes, pad=0.025, fraction=0.075, shrink=0.5,
                          anchor=(-1, 0.5))
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        plt.setp(cb_yticks, color=font_color)
    axes.axis('off')

    my_topo_plot = _iter_topography(info, layout=layout, on_pick=on_pick,
                                    fig=fig, layout_scale=layout_scale,
                                    axis_spinecolor=border,
                                    axis_facecolor=axis_facecolor,
                                    fig_facecolor=fig_facecolor,
                                    unified=unified, img=img, axes=axes)

    for ax, ch_idx in my_topo_plot:
        if layout.kind == 'Vectorview-all' and ylim is not None:
            this_type = {'mag': 0, 'grad': 1}[channel_type(info, ch_idx)]
            ylim_ = [v[this_type] if _check_vlim(v) else v for v in ylim]
        else:
            ylim_ = ylim

        show_func(ax, ch_idx, tmin=tmin, tmax=tmax, vmin=vmin,
                  vmax=vmax, ylim=ylim_)

    if title is not None:
        plt.figtext(0.03, 0.95, title, color=font_color, fontsize=15, va='top')

    return fig


def _plot_topo_onpick(event, show_func):
    """Onpick callback that shows a single channel in a new figure."""
    # make sure that the swipe gesture in OS-X doesn't open many figures
    orig_ax = event.inaxes
    import matplotlib.pyplot as plt
    try:
        if hasattr(orig_ax, '_mne_axs'):  # in unified, single-axes mode
            x, y = event.xdata, event.ydata
            for ax in orig_ax._mne_axs:
                if x >= ax.pos[0] and y >= ax.pos[1] and \
                        x <= ax.pos[0] + ax.pos[2] and \
                        y <= ax.pos[1] + ax.pos[3]:
                    orig_ax = ax
                    break
            else:
                # no axis found
                return
        elif not hasattr(orig_ax, '_mne_ch_idx'):
            # neither old nor new mode
            return
        ch_idx = orig_ax._mne_ch_idx
        face_color = orig_ax._mne_ax_face_color
        fig, ax = plt.subplots(1)

        plt.title(orig_ax._mne_ch_name)
        ax.set_facecolor(face_color)

        # allow custom function to override parameters
        show_func(ax, ch_idx)
        plt_show(fig=fig)

    except Exception as err:
        # matplotlib silently ignores exceptions in event handlers,
        # so we print
        # it here to know what went wrong
        print(err)
        raise


def _compute_ax_scalings(bn, xlim, ylim):
    """Compute scale factors for a unified plot."""
    if isinstance(ylim[0], (tuple, list, np.ndarray)):
        ylim = (ylim[0][0], ylim[1][0])
    pos = bn.pos
    bn.x_s = pos[2] / (xlim[1] - xlim[0])
    bn.x_t = pos[0] - bn.x_s * xlim[0]
    bn.y_s = pos[3] / (ylim[1] - ylim[0])
    bn.y_t = pos[1] - bn.y_s * ylim[0]


def _check_vlim(vlim):
    """Check the vlim."""
    return not np.isscalar(vlim) and vlim is not None


def _imshow_tfr(ax, ch_idx, tmin, tmax, vmin, vmax, onselect, ylim=None,
                tfr=None, freq=None, x_label=None, y_label=None,
                colorbar=False, cmap=('RdBu_r', True), yscale='auto',
                mask=None, mask_style="both", mask_cmap="Greys",
                mask_alpha=0.1, is_jointplot=False, cnorm=None):
    """Show time-frequency map as two-dimensional image."""
    from matplotlib import pyplot as plt
    from matplotlib.widgets import RectangleSelector

    _check_option('yscale', yscale, ['auto', 'linear', 'log'])

    cmap, interactive_cmap = cmap
    times = np.linspace(tmin, tmax, num=tfr[ch_idx].shape[1])

    img, t_end = _plot_masked_image(
        ax, tfr[ch_idx], times, mask, yvals=freq, cmap=cmap,
        vmin=vmin, vmax=vmax, mask_style=mask_style, mask_alpha=mask_alpha,
        mask_cmap=mask_cmap, yscale=yscale, cnorm=cnorm)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if colorbar:
        if isinstance(colorbar, DraggableColorbar):
            cbar = colorbar.cbar  # this happens with multiaxes case
        else:
            cbar = plt.colorbar(mappable=img, ax=ax)
        if interactive_cmap:
            ax.CB = DraggableColorbar(cbar, img)
    ax.RS = RectangleSelector(ax, onselect=onselect)  # reference must be kept

    return t_end


def _imshow_tfr_unified(bn, ch_idx, tmin, tmax, vmin, vmax, onselect,
                        ylim=None, tfr=None, freq=None, vline=None,
                        x_label=None, y_label=None, colorbar=False,
                        picker=True, cmap='RdBu_r', title=None, hline=None):
    """Show multiple tfrs on topo using a single axes."""
    _compute_ax_scalings(bn, (tmin, tmax), (freq[0], freq[-1]))
    ax = bn.ax
    data_lines = bn.data_lines
    extent = (bn.x_t + bn.x_s * tmin, bn.x_t + bn.x_s * tmax,
              bn.y_t + bn.y_s * freq[0], bn.y_t + bn.y_s * freq[-1])
    data_lines.append(ax.imshow(tfr[ch_idx], clip_on=True, clip_box=bn.pos,
                                extent=extent, aspect="auto", origin="lower",
                                vmin=vmin, vmax=vmax, cmap=cmap))


def _plot_timeseries(ax, ch_idx, tmin, tmax, vmin, vmax, ylim, data, color,
                     times, vline=None, x_label=None, y_label=None,
                     colorbar=False, hline=None, hvline_color='w',
                     labels=None):
    """Show time series on topo split across multiple axes."""
    import matplotlib.pyplot as plt
    picker_flag = False
    for data_, color_, times_ in zip(data, color, times):
        if not picker_flag:
            # use large tol for picker so we can click anywhere in the axes
            line = ax.plot(times_, data_[ch_idx], color=color_, picker=True)[0]
            line.set_pickradius(1e9)
            picker_flag = True
        else:
            ax.plot(times_, data_[ch_idx], color=color_)

    def _format_coord(x, y, labels, ax):
        """Create status string based on cursor coordinates."""
        # find indices for datasets near cursor (if any)
        tdiffs = [np.abs(tvec - x).min() for tvec in times]
        nearby = [k for k, tdiff in enumerate(tdiffs) if
                  tdiff < (tmax - tmin) / 100]
        xlabel = ax.get_xlabel()
        xunit = (xlabel[xlabel.find('(') + 1:xlabel.find(')')]
                 if '(' in xlabel and ')' in xlabel else 's')
        timestr = '%6.3f %s: ' % (x, xunit)
        if not nearby:
            return '%s Nothing here' % timestr
        labels = [''] * len(nearby) if labels is None else labels
        nearby_data = [(data[n], labels[n], times[n]) for n in nearby]
        ylabel = ax.get_ylabel()
        yunit = (ylabel[ylabel.find('(') + 1:ylabel.find(')')]
                 if '(' in ylabel and ')' in ylabel else '')
        # try to estimate whether to truncate condition labels
        slen = 9 + len(xunit) + sum([12 + len(yunit) + len(label)
                                     for label in labels])
        bar_width = (ax.figure.get_size_inches() * ax.figure.dpi)[0] / 5.5
        # show labels and y values for datasets near cursor
        trunc_labels = bar_width < slen
        s = timestr
        for data_, label, tvec in nearby_data:
            idx = np.abs(tvec - x).argmin()
            s += '%7.2f %s' % (data_[ch_idx, idx], yunit)
            if trunc_labels:
                label = (label if len(label) <= 10 else
                         '%s..%s' % (label[:6], label[-2:]))
            s += ' [%s] ' % label if label else ' '
        return s
    ax.format_coord = lambda x, y: _format_coord(x, y, labels=labels, ax=ax)

    def _cursor_vline(event):
        """Draw cursor (vertical line)."""
        ax = event.inaxes
        if not ax:
            return
        if ax._cursorline is not None:
            ax._cursorline.remove()
        ax._cursorline = ax.axvline(event.xdata, color=ax._cursorcolor)
        ax.figure.canvas.draw()

    def _rm_cursor(event):
        ax = event.inaxes
        if ax._cursorline is not None:
            ax._cursorline.remove()
            ax._cursorline = None
        ax.figure.canvas.draw()

    ax._cursorline = None
    # choose cursor color based on perceived brightness of background
    facecol = _to_rgb(ax.get_facecolor())
    face_brightness = np.dot(facecol, [299, 587, 114])
    ax._cursorcolor = 'white' if face_brightness < 150 else 'black'

    plt.connect('motion_notify_event', _cursor_vline)
    plt.connect('axes_leave_event', _rm_cursor)

    ymin, ymax = ax.get_ylim()
    # don't pass vline or hline here (this fxn doesn't do hvline_color):
    _setup_ax_spines(ax, [], tmin, tmax, ymin, ymax, hline=False)
    ax.figure.set_facecolor('k' if hvline_color == 'w' else 'w')
    ax.spines['bottom'].set_color(hvline_color)
    ax.spines['left'].set_color(hvline_color)
    ax.tick_params(axis='x', colors=hvline_color, which='both')
    ax.tick_params(axis='y', colors=hvline_color, which='both')
    ax.title.set_color(hvline_color)
    ax.xaxis.label.set_color(hvline_color)
    ax.yaxis.label.set_color(hvline_color)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        if isinstance(y_label, list):
            ax.set_ylabel(y_label[ch_idx])
        else:
            ax.set_ylabel(y_label)

    if vline:
        plt.axvline(vline, color=hvline_color, linewidth=1.0,
                    linestyle='--')
    if hline:
        plt.axhline(hline, color=hvline_color, linewidth=1.0, zorder=10)

    if colorbar:
        plt.colorbar()


def _plot_timeseries_unified(bn, ch_idx, tmin, tmax, vmin, vmax, ylim, data,
                             color, times, vline=None, x_label=None,
                             y_label=None, colorbar=False, hline=None,
                             hvline_color='w'):
    """Show multiple time series on topo using a single axes."""
    import matplotlib.pyplot as plt
    if not (ylim and not any(v is None for v in ylim)):
        ylim = [min(np.min(d) for d in data), max(np.max(d) for d in data)]
    # Translation and scale parameters to take data->under_ax normalized coords
    _compute_ax_scalings(bn, (tmin, tmax), ylim)
    pos = bn.pos
    data_lines = bn.data_lines
    ax = bn.ax
    # XXX These calls could probably be made faster by using collections
    for data_, color_, times_ in zip(data, color, times):
        data_lines.append(ax.plot(
            bn.x_t + bn.x_s * times_, bn.y_t + bn.y_s * data_[ch_idx],
            linewidth=0.5, color=color_, clip_on=True, clip_box=pos)[0])
    if vline:
        vline = np.array(vline) * bn.x_s + bn.x_t
        ax.vlines(vline, pos[1], pos[1] + pos[3], color=hvline_color,
                  linewidth=0.5, linestyle='--')
    if hline:
        hline = np.array(hline) * bn.y_s + bn.y_t
        ax.hlines(hline, pos[0], pos[0] + pos[2], color=hvline_color,
                  linewidth=0.5)
    if x_label is not None:
        ax.text(pos[0] + pos[2] / 2., pos[1], x_label,
                horizontalalignment='center', verticalalignment='top')
    if y_label is not None:
        y_label = y_label[ch_idx] if isinstance(y_label, list) else y_label
        ax.text(pos[0], pos[1] + pos[3] / 2., y_label,
                horizontalignment='right', verticalalignment='middle',
                rotation=90)
    if colorbar:
        plt.colorbar()


def _erfimage_imshow(ax, ch_idx, tmin, tmax, vmin, vmax, ylim=None, data=None,
                     epochs=None, sigma=None, order=None, scalings=None,
                     vline=None, x_label=None, y_label=None, colorbar=False,
                     cmap='RdBu_r', vlim_array=None):
    """Plot erfimage on sensor topography."""
    from scipy import ndimage
    import matplotlib.pyplot as plt
    this_data = data[:, ch_idx, :]
    if vlim_array is not None:
        vmin, vmax = vlim_array[ch_idx]

    if callable(order):
        order = order(epochs.times, this_data)

    if order is not None:
        this_data = this_data[order]

    if sigma > 0.:
        this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

    img = ax.imshow(this_data, extent=[tmin, tmax, 0, len(data)],
                    aspect='auto', origin='lower', vmin=vmin, vmax=vmax,
                    picker=True, cmap=cmap, interpolation='nearest')

    ax = plt.gca()
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if colorbar:
        plt.colorbar(mappable=img)


def _erfimage_imshow_unified(bn, ch_idx, tmin, tmax, vmin, vmax, ylim=None,
                             data=None, epochs=None, sigma=None, order=None,
                             scalings=None, vline=None, x_label=None,
                             y_label=None, colorbar=False, cmap='RdBu_r',
                             vlim_array=None):
    """Plot erfimage topography using a single axis."""
    from scipy import ndimage
    _compute_ax_scalings(bn, (tmin, tmax), (0, len(epochs.events)))
    ax = bn.ax
    data_lines = bn.data_lines
    extent = (bn.x_t + bn.x_s * tmin, bn.x_t + bn.x_s * tmax, bn.y_t,
              bn.y_t + bn.y_s * len(epochs.events))
    this_data = data[:, ch_idx, :]
    vmin, vmax = (None, None) if vlim_array is None else vlim_array[ch_idx]

    if callable(order):
        order = order(epochs.times, this_data)

    if order is not None:
        this_data = this_data[order]

    if sigma > 0.:
        this_data = ndimage.gaussian_filter1d(this_data, sigma=sigma, axis=0)

    data_lines.append(ax.imshow(this_data, extent=extent, aspect='auto',
                                origin='lower', vmin=vmin, vmax=vmax,
                                picker=True, cmap=cmap,
                                interpolation='nearest'))


def _plot_evoked_topo(evoked, layout=None, layout_scale=0.945,
                      color=None, border='none', ylim=None, scalings=None,
                      title=None, proj=False, vline=(0.,), hline=(0.,),
                      fig_facecolor='k', fig_background=None,
                      axis_facecolor='k', font_color='w', merge_channels=False,
                      legend=True, axes=None, exclude='bads', show=True,
                      noise_cov=None):
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
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas.
    color : list of color objects | color object | None
        Everything matplotlib accepts to specify colors. If not list-like,
        the color specified will be repeated. If None, colors are
        automatically drawn.
    border : str
        Matplotlib borders style to be used for each sensor plot.
    ylim : dict | None
        ylim for plots (after scaling has been applied). The value
        determines the upper and lower subplot limits. e.g.
        ylim = dict(eeg=[-20, 20]). Valid keys are eeg, mag, grad. If None,
        the ylim parameter for each channel type is determined by the minimum
        and maximum peak.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
    title : str
        Title of the figure.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    vline : list of floats | None
        The values at which to show a vertical line.
    hline : list of floats | None
        The values at which to show a horizontal line.
    fig_facecolor : color
        The figure face color. Defaults to black.
    fig_background : None | array
        A background image for the figure. This must be a valid input to
        `matplotlib.pyplot.imshow`. Defaults to None.
    axis_facecolor : color
        The face color to be used for each sensor plot. Defaults to black.
    font_color : color
        The color of text in the colorbar and title. Defaults to white.
    merge_channels : bool
        Whether to use RMS value of gradiometer pairs. Only works for Neuromag
        data. Defaults to False.
    legend : bool | int | string | tuple
        If True, create a legend based on evoked.comment. If False, disable the
        legend. Otherwise, the legend is created and the parameter value is
        passed as the location parameter to the matplotlib legend call. It can
        be an integer (e.g. 0 corresponds to upper right corner of the plot),
        a string (e.g. 'upper right'), or a tuple (x, y coordinates of the
        lower left corner of the legend in the axes coordinate system).
        See matplotlib documentation for more details.
    axes : instance of matplotlib Axes | None
        Axes to plot into. If None, axes will be created.
    noise_cov : instance of Covariance | str | None
        Noise covariance used to whiten the data while plotting.
        Whitened data channels names are shown in italic.
        Can be a string to load a covariance from disk.
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded. By default, exclude is set to 'bads'.
    show : bool
        Show figure if True.

        .. versionadded:: 0.16.0

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Images of evoked responses at sensor locations
    """
    import matplotlib.pyplot as plt
    from ..cov import whiten_evoked

    if not type(evoked) in (tuple, list):
        evoked = [evoked]

    if type(color) in (tuple, list):
        if len(color) != len(evoked):
            raise ValueError('Lists of evoked objects and colors'
                             ' must have the same length')
    elif color is None:
        colors = ['w'] + _get_color_list
        stop = (slice(len(evoked)) if len(evoked) < len(colors)
                else slice(len(colors)))
        color = cycle(colors[stop])
        if len(evoked) > len(colors):
            warn('More evoked objects than colors available. You should pass '
                 'a list of unique colors.')
    else:
        color = cycle([color])

    noise_cov = _check_cov(noise_cov, evoked[0].info)
    if noise_cov is not None:
        evoked = [whiten_evoked(e, noise_cov) for e in evoked]
    else:
        evoked = [e.copy() for e in evoked]
    info = evoked[0].info
    ch_names = evoked[0].ch_names
    scalings = _handle_default('scalings', scalings)
    if not all(e.ch_names == ch_names for e in evoked):
        raise ValueError('All evoked.picks must be the same')
    ch_names = _clean_names(ch_names)
    if merge_channels:
        picks = _pair_grad_sensors(info, topomap_coords=False, exclude=exclude)
        chs = list()
        for pick in picks[::2]:
            ch = info['chs'][pick]
            ch['ch_name'] = ch['ch_name'][:-1] + 'X'
            chs.append(ch)
        with info._unlock(update_redundant=True, check_after=True):
            info['chs'] = chs
            info['bads'] = list()   # Bads handled by pair_grad_sensors
        new_picks = list()
        for e in evoked:
            data, _ = _merge_ch_data(e.data[picks], 'grad', [])
            if noise_cov is None:
                data *= scalings['grad']
            e.data = data
            new_picks.append(range(len(data)))
        picks = new_picks
        types_used = ['grad']
        unit = _handle_default('units')['grad'] if noise_cov is None else 'NA'
        y_label = 'RMS amplitude (%s)' % unit

    if layout is None:
        layout = find_layout(info, exclude=exclude)

    if not merge_channels:
        # XXX. at the moment we are committed to 1- / 2-sensor-types layouts
        chs_in_layout = [ch_name for ch_name in ch_names
                         if ch_name in layout.names]
        types_used = [channel_type(info, ch_names.index(ch))
                      for ch in chs_in_layout]
        # Using dict conversion to remove duplicates
        types_used = list(dict.fromkeys(types_used))
        # remove possible reference meg channels
        types_used = [types_used for types_used in types_used
                      if types_used != 'ref_meg']
        # one check for all vendors
        is_meg = len([x for x in types_used if x in ['mag', 'grad']]) > 0
        is_nirs = len([x for x in types_used if x in
                      ('hbo', 'hbr', 'fnirs_cw_amplitude', 'fnirs_od')]) > 0
        if is_meg:
            types_used = list(types_used)[::-1]  # -> restore kwarg order
            picks = [pick_types(info, meg=kk, ref_meg=False, exclude=exclude)
                     for kk in types_used]
        elif is_nirs:
            types_used = list(types_used)[::-1]  # -> restore kwarg order
            picks = [pick_types(info, fnirs=kk, ref_meg=False, exclude=exclude)
                     for kk in types_used]
        else:
            types_used_kwargs = {t: True for t in types_used}
            picks = [pick_types(info, meg=False, exclude=exclude,
                                **types_used_kwargs)]
        assert isinstance(picks, list) and len(types_used) == len(picks)

        if noise_cov is None:
            for e in evoked:
                for pick, ch_type in zip(picks, types_used):
                    e.data[pick] *= scalings[ch_type]

        if proj is True and all(e.proj is not True for e in evoked):
            evoked = [e.apply_proj() for e in evoked]
        elif proj == 'interactive':  # let it fail early.
            for e in evoked:
                _check_delayed_ssp(e)
        # Y labels for picked plots must be reconstructed
        y_label = list()
        for ch_idx in range(len(chs_in_layout)):
            if noise_cov is None:
                unit = _handle_default('units')[channel_type(info, ch_idx)]
            else:
                unit = 'NA'
            y_label.append('Amplitude (%s)' % unit)

    if ylim is None:
        # find minima and maxima over all evoked data for each channel pick
        ymaxes = np.array([max((e.data[t]).max() for e in evoked)
                           for t in picks])
        ymins = np.array([min((e.data[t]).min() for e in evoked)
                          for t in picks])

        ylim_ = (ymins, ymaxes)
    elif isinstance(ylim, dict):
        ylim_ = _handle_default('ylim', ylim)
        ylim_ = [ylim_[kk] for kk in types_used]
        # extra unpack to avoid bug #1700
        if len(ylim_) == 1:
            ylim_ = ylim_[0]
        else:
            ylim_ = [np.array(yl) for yl in ylim_]
            # Transposing to avoid Zipping confusion
            if is_meg or is_nirs:
                ylim_ = list(map(list, zip(*ylim_)))
    else:
        raise TypeError('ylim must be None or a dict. Got %s.' % type(ylim))

    data = [e.data for e in evoked]
    comments = [e.comment for e in evoked]
    times = [e.times for e in evoked]

    show_func = partial(_plot_timeseries_unified, data=data, color=color,
                        times=times, vline=vline, hline=hline,
                        hvline_color=font_color)
    click_func = partial(_plot_timeseries, data=data, color=color, times=times,
                         vline=vline, hline=hline, hvline_color=font_color,
                         labels=comments)

    time_min = min([t[0] for t in times])
    time_max = max([t[-1] for t in times])
    fig = _plot_topo(info=info, times=[time_min, time_max],
                     show_func=show_func, click_func=click_func, layout=layout,
                     colorbar=False, ylim=ylim_, cmap=None,
                     layout_scale=layout_scale, border=border,
                     fig_facecolor=fig_facecolor, font_color=font_color,
                     axis_facecolor=axis_facecolor, title=title,
                     x_label='Time (s)', y_label=y_label, unified=True,
                     axes=axes)

    add_background_image(fig, fig_background)

    if legend is not False:
        legend_loc = 0 if legend is True else legend
        labels = [e.comment if e.comment else 'Unknown' for e in evoked]
        legend = plt.legend(labels, loc=legend_loc,
                            prop={'size': 10})
        legend.get_frame().set_facecolor(axis_facecolor)
        txts = legend.get_texts()
        for txt, col in zip(txts, color):
            txt.set_color(col)

    if proj == 'interactive':
        for e in evoked:
            _check_delayed_ssp(e)
        params = dict(evokeds=evoked, times=times,
                      plot_update_proj_callback=_plot_update_evoked_topo_proj,
                      projs=evoked[0].info['projs'], fig=fig)
        _draw_proj_checkbox(None, params)

    plt_show(show)
    return fig


def _plot_update_evoked_topo_proj(params, bools):
    """Update topo sensor plots."""
    evokeds = [e.copy() for e in params['evokeds']]
    fig = params['fig']
    projs = [proj for proj, b in zip(params['projs'], bools) if b]
    params['proj_bools'] = bools
    for e in evokeds:
        e.add_proj(projs, remove_existing=True)
        e.apply_proj()

    # make sure to only modify the time courses, not the ticks
    for ax in fig.axes[0]._mne_axs:
        for line, evoked in zip(ax.data_lines, evokeds):
            line.set_ydata(ax.y_t + ax.y_s * evoked.data[ax._mne_ch_idx])

    fig.canvas.draw()


def plot_topo_image_epochs(epochs, layout=None, sigma=0., vmin=None,
                           vmax=None, colorbar=None, order=None, cmap='RdBu_r',
                           layout_scale=.95, title=None, scalings=None,
                           border='none', fig_facecolor='k',
                           fig_background=None, font_color='w', show=True):
    """Plot Event Related Potential / Fields image on topographies.

    Parameters
    ----------
    epochs : instance of :class:`~mne.Epochs`
        The epochs.
    layout : instance of Layout
        System specific sensor positions.
    sigma : float
        The standard deviation of the Gaussian smoothing to apply along
        the epoch axis to apply in the image. If 0., no smoothing is applied.
    vmin : float
        The min value in the image. The unit is µV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    vmax : float
        The max value in the image. The unit is µV for EEG channels,
        fT for magnetometers and fT/cm for gradiometers.
    colorbar : bool | None
        Whether to display a colorbar or not. If ``None`` a colorbar will be
        shown only if all channels are of the same type. Defaults to ``None``.
    order : None | array of int | callable
        If not None, order is used to reorder the epochs on the y-axis
        of the image. If it's an array of int it should be of length
        the number of good epochs. If it's a callable the arguments
        passed are the times vector and the data as 2d array
        (data.shape[1] == len(times)).
    cmap : colormap
        Colors to be mapped to the values.
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas.
    title : str
        Title of the figure.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If
        ``None``, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
    border : str
        Matplotlib borders style to be used for each sensor plot.
    fig_facecolor : color
        The figure face color. Defaults to black.
    fig_background : None | array
        A background image for the figure. This must be a valid input to
        :func:`matplotlib.pyplot.imshow`. Defaults to ``None``.
    font_color : color
        The color of tick labels in the colorbar. Defaults to white.
    show : bool
        Whether to show the figure. Defaults to ``True``.

    Returns
    -------
    fig : instance of :class:`matplotlib.figure.Figure`
        Figure distributing one image per channel across sensor topography.

    Notes
    -----
    In an interactive Python session, this plot will be interactive; clicking
    on a channel image will pop open a larger view of the image; this image
    will always have a colorbar even when the topo plot does not (because it
    shows multiple sensor types).
    """
    scalings = _handle_default('scalings', scalings)

    # make a copy because we discard non-data channels and scale the data
    epochs = epochs.copy().load_data()
    # use layout to subset channels present in epochs object
    if layout is None:
        layout = find_layout(epochs.info)
    ch_names = set(layout.names) & set(epochs.ch_names)
    idxs = [epochs.ch_names.index(ch_name) for ch_name in ch_names]
    epochs = epochs.pick(idxs)
    # get lists of channel type & scale coefficient
    ch_types = epochs.get_channel_types()
    scale_coeffs = [scalings.get(ch_type, 1) for ch_type in ch_types]
    # scale the data
    epochs._data *= np.array(scale_coeffs)[:, np.newaxis]
    data = epochs.get_data()
    # get vlims for each channel type
    vlim_dict = dict()
    for ch_type in set(ch_types):
        this_data = data[:, np.where(np.array(ch_types) == ch_type)]
        vlim_dict[ch_type] = _setup_vmin_vmax(this_data, vmin, vmax)
    vlim_array = np.array([vlim_dict[ch_type] for ch_type in ch_types])
    # only show colorbar if we have a single channel type
    if colorbar is None:
        colorbar = (len(set(ch_types)) == 1)
    # if colorbar=True, we know we have only 1 channel type so all entries
    # in vlim_array are the same, just take the first one
    if colorbar and vmin is None and vmax is None:
        vmin, vmax = vlim_array[0]

    show_func = partial(_erfimage_imshow_unified, scalings=scale_coeffs,
                        order=order, data=data, epochs=epochs, sigma=sigma,
                        cmap=cmap, vlim_array=vlim_array)

    erf_imshow = partial(_erfimage_imshow, scalings=scale_coeffs, order=order,
                         data=data, epochs=epochs, sigma=sigma, cmap=cmap,
                         vlim_array=vlim_array, colorbar=True)

    fig = _plot_topo(info=epochs.info, times=epochs.times,
                     click_func=erf_imshow, show_func=show_func, layout=layout,
                     colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                     layout_scale=layout_scale, title=title,
                     fig_facecolor=fig_facecolor, font_color=font_color,
                     border=border, x_label='Time (s)', y_label='Epoch',
                     unified=True, img=True)
    add_background_image(fig, fig_background)
    plt_show(show)
    return fig
