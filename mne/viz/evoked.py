# -*- coding: utf-8 -*-
"""Functions to plot evoked M/EEG data (besides topographies)."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

from copy import deepcopy
from functools import partial
from numbers import Integral
from typing import Iterable

import numpy as np

from ..io.pick import (channel_type, _pick_data_channels,
                       _VALID_CHANNEL_TYPES, channel_indices_by_type,
                       _DATA_CH_TYPES_SPLIT, _pick_inst, _get_channel_types,
                       _PICK_TYPES_DATA_DICT, _picks_to_idx, pick_info)
from ..defaults import _handle_default
from .utils import (_draw_proj_checkbox, tight_layout, _check_delayed_ssp,
                    plt_show, _process_times, DraggableColorbar, _setup_cmap,
                    _setup_vmin_vmax, _grad_pair_pick_and_name, _check_cov,
                    _validate_if_list_of_axes, _triage_rank_sss,
                    _connection_line, _get_color_list, _setup_ax_spines,
                    _setup_plot_projector, _prepare_joint_axes,
                    _set_title_multiple_electrodes, _check_time_unit,
                    _plot_masked_image)
from ..utils import (logger, _clean_names, warn, _pl, verbose, _validate_type,
                     _check_if_nan, _check_ch_locs, fill_doc)

from .topo import _plot_evoked_topo
from .topomap import (_prepare_topo_plot, plot_topomap, _check_outlines,
                      _draw_outlines, _prepare_topomap, _set_contour_locator)
from ..channels.layout import (_pair_grad_sensors, _auto_topomap_coords,
                               find_layout)


def _butterfly_onpick(event, params):
    """Add a channel name on click."""
    params['need_draw'] = True
    ax = event.artist.axes
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
    text.set_zorder(len(ax.lines))  # to make sure it goes on top of the lines
    text.set_path_effects(params['path_effects'])
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use on_button_press (happens once per click)
    # to do the drawing


def _butterfly_on_button_press(event, params):
    """Only draw once for picking."""
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


def _line_plot_onselect(xmin, xmax, ch_types, info, data, times, text=None,
                        psd=False, time_unit='s'):
    """Draw topomaps from the selected area."""
    import matplotlib.pyplot as plt
    ch_types = [type_ for type_ in ch_types if type_ in ('eeg', 'grad', 'mag')]
    if len(ch_types) == 0:
        raise ValueError('Interactive topomaps only allowed for EEG '
                         'and MEG channels.')
    if ('grad' in ch_types and
            len(_pair_grad_sensors(info, topomap_coords=False,
                                   raise_error=False)) < 2):
        ch_types.remove('grad')
        if len(ch_types) == 0:
            return

    vert_lines = list()
    if text is not None:
        text.set_visible(True)
        ax = text.axes
        vert_lines.append(ax.axvline(xmin, zorder=0, color='red'))
        vert_lines.append(ax.axvline(xmax, zorder=0, color='red'))
        fill = ax.axvspan(xmin, xmax, alpha=0.2, color='green')
        evoked_fig = plt.gcf()
        evoked_fig.canvas.draw()
        evoked_fig.canvas.flush_events()

    minidx = np.abs(times - xmin).argmin()
    maxidx = np.abs(times - xmax).argmin()
    fig, axarr = plt.subplots(1, len(ch_types), squeeze=False,
                              figsize=(3 * len(ch_types), 3))

    for idx, ch_type in enumerate(ch_types):
        if ch_type not in ('eeg', 'grad', 'mag'):
            continue
        picks, pos, merge_grads, _, ch_type = _prepare_topo_plot(
            info, ch_type, layout=None)
        if len(pos) < 2:
            fig.delaxes(axarr[0][idx])
            continue
        this_data = data[picks, minidx:maxidx]
        if merge_grads:
            from ..channels.layout import _merge_grad_data
            method = 'mean' if psd else 'rms'
            this_data = _merge_grad_data(this_data, method=method)
            title = '%s %s' % (ch_type, method.upper())
        else:
            title = ch_type
        this_data = np.average(this_data, axis=1)
        axarr[0][idx].set_title(title)
        vmin = min(this_data) if psd else None
        vmax = max(this_data) if psd else None  # All negative for dB psd.
        cmap = 'Reds' if psd else None
        plot_topomap(this_data, pos, cmap=cmap, vmin=vmin, vmax=vmax,
                     axes=axarr[0][idx], show=False)

    unit = 'Hz' if psd else time_unit
    fig.suptitle('Average over %.2f%s - %.2f%s' % (xmin, unit, xmax, unit),
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
    """Remove lines from evoked plot as topomap is closed."""
    for line in lines:
        ax.lines.remove(line)
    ax.patches.remove(fill)
    ax.get_figure().canvas.draw()


def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb


def _plot_legend(pos, colors, axis, bads, outlines, loc, size=30):
    """Plot (possibly colorized) channel legends for evoked plots."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axis.get_figure().canvas.draw()
    bbox = axis.get_window_extent()  # Determine the correct size.
    ratio = bbox.width / bbox.height
    ax = inset_axes(axis, width=str(size / ratio) + '%',
                    height=str(size) + '%', loc=loc)
    ax.set_adjustable("box")
    _prepare_topomap(pos, ax, check_nonzero=False)
    pos_x, pos_y = pos.T
    ax.scatter(pos_x, pos_y, color=colors, s=size * .8, marker='.', zorder=1)
    if bads:
        bads = np.array(bads)
        ax.scatter(pos_x[bads], pos_y[bads], s=size / 6, marker='.',
                   color='w', zorder=1)
    _draw_outlines(ax, outlines)


def _plot_evoked(evoked, picks, exclude, unit, show, ylim, proj, xlim, hline,
                 units, scalings, titles, axes, plot_type, cmap=None,
                 gfp=False, window_title=None, spatial_colors=False,
                 set_tight_layout=True, selectable=True, zorder='unsorted',
                 noise_cov=None, colorbar=True, mask=None, mask_style=None,
                 mask_cmap=None, mask_alpha=.25, time_unit='s',
                 show_names=False, group_by=None):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings).

    Extra param is:

    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    """
    import matplotlib.pyplot as plt

    # For evoked.plot_image ...
    # First input checks for group_by and axes if any of them is not None.
    # Either both must be dicts, or neither.
    # If the former, the two dicts provide picks and axes to plot them to.
    # Then, we call this function recursively for each entry in `group_by`.
    if plot_type == "image" and isinstance(group_by, dict):
        if axes is None:
            axes = dict()
            for sel in group_by:
                plt.figure()
                axes[sel] = plt.axes()
        if not isinstance(axes, dict):
            raise ValueError("If `group_by` is a dict, `axes` must be "
                             "a dict of axes or None.")
        _validate_if_list_of_axes(list(axes.values()))
        remove_xlabels = any([ax.is_last_row() for ax in axes.values()])
        for sel in group_by:  # ... we loop over selections
            if sel not in axes:
                raise ValueError(sel + " present in `group_by`, but not "
                                 "found in `axes`")
            ax = axes[sel]
            # the unwieldy dict comp below defaults the title to the sel
            _plot_evoked(evoked, group_by[sel], exclude, unit, show, ylim,
                         proj, xlim, hline, units, scalings,
                         (titles if titles is not None else
                          {channel_type(evoked.info, idx): sel
                           for idx in group_by[sel]}),
                         ax, plot_type, cmap=cmap, gfp=gfp,
                         window_title=window_title,
                         set_tight_layout=set_tight_layout,
                         selectable=selectable, noise_cov=noise_cov,
                         colorbar=colorbar, mask=mask,
                         mask_style=mask_style, mask_cmap=mask_cmap,
                         mask_alpha=mask_alpha, time_unit=time_unit,
                         show_names=show_names)
            if remove_xlabels and not ax.is_last_row():
                ax.set_xticklabels([])
                ax.set_xlabel("")
        ims = [ax.images[0] for ax in axes.values()]
        clims = np.array([im.get_clim() for im in ims])
        min, max = clims.min(), clims.max()
        for im in ims:
            im.set_clim(min, max)
        figs = [ax.get_figure() for ax in axes.values()]
        if len(set(figs)) == 1:
            return figs[0]
        else:
            return figs
    elif isinstance(axes, dict):
        raise ValueError("If `group_by` is not a dict, "
                         "`axes` must not be a dict either.")

    time_unit, times = _check_time_unit(time_unit, evoked.times)
    info = evoked.info
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')
    if isinstance(gfp, str) and gfp != 'only':
        raise ValueError('gfp must be boolean or "only". Got %s' % gfp)

    scalings = _handle_default('scalings', scalings)
    titles = _handle_default('titles', titles)
    units = _handle_default('units', units)

    picks = _picks_to_idx(info, picks, none='all', exclude=())
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")

    bad_ch_idx = [info['ch_names'].index(ch) for ch in info['bads']
                  if ch in info['ch_names']]
    if len(exclude) > 0:
        if isinstance(exclude, str) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list) and
              all(isinstance(ch, str) for ch in exclude)):
            exclude = [info['ch_names'].index(ch) for ch in exclude]
        else:
            raise ValueError(
                'exclude has to be a list of channel names or "bads"')

        picks = np.array([pick for pick in picks if pick not in exclude])

    types = np.array([channel_type(info, idx) for idx in picks], np.unicode)
    ch_types_used = list()
    for this_type in _VALID_CHANNEL_TYPES:
        if this_type in types:
            ch_types_used.append(this_type)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(ch_types_used), 1)
        plt.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)
        if isinstance(axes, plt.Axes):
            axes = [axes]
        fig.set_size_inches(6.4, 2 + len(axes))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if fig is None:
        fig = axes[0].get_figure()

    if window_title is not None:
        fig.canvas.set_window_title(window_title)

    if len(axes) != len(ch_types_used):
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%d: %s)' % (len(axes), len(ch_types_used),
                                             sorted(ch_types_used)))
    noise_cov = _check_cov(noise_cov, info)
    projector, whitened_ch_names = _setup_plot_projector(
        info, noise_cov, proj=proj is True, nave=evoked.nave)
    evoked = evoked.copy()
    if len(whitened_ch_names) > 0:
        unit = False
    if projector is not None:
        evoked.data[:] = np.dot(projector, evoked.data)
    if plot_type == 'butterfly':
        _plot_lines(evoked.data, info, picks, fig, axes, spatial_colors, unit,
                    units, scalings, hline, gfp, types, zorder, xlim, ylim,
                    times, bad_ch_idx, titles, ch_types_used, selectable,
                    False, line_alpha=1., nave=evoked.nave,
                    time_unit=time_unit)
        plt.setp(axes, xlabel='Time (%s)' % time_unit)

    elif plot_type == 'image':
        for ai, (ax, this_type) in enumerate(zip(axes, ch_types_used)):
            use_nave = evoked.nave if ai == 0 else None
            this_picks = list(picks[types == this_type])
            _plot_image(evoked.data, ax, this_type, this_picks, cmap, unit,
                        units, scalings, times, xlim, ylim, titles,
                        colorbar=colorbar, mask=mask, mask_style=mask_style,
                        mask_cmap=mask_cmap, mask_alpha=mask_alpha,
                        nave=use_nave, time_unit=time_unit,
                        show_names=show_names, ch_names=evoked.ch_names)
    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=info['projs'], axes=axes,
                      types=types, units=units, scalings=scalings, unit=unit,
                      ch_types_used=ch_types_used, picks=picks,
                      plot_update_proj_callback=_plot_update_evoked,
                      plot_type=plot_type)
        _draw_proj_checkbox(None, params)

    plt.setp(fig.axes[:len(ch_types_used) - 1], xlabel='')
    fig.canvas.draw()  # for axes plots update axes.
    if set_tight_layout:
        tight_layout(fig=fig)
    plt_show(show)
    return fig


def _plot_lines(data, info, picks, fig, axes, spatial_colors, unit, units,
                scalings, hline, gfp, types, zorder, xlim, ylim, times,
                bad_ch_idx, titles, ch_types_used, selectable, psd,
                line_alpha, nave, time_unit='ms'):
    """Plot data as butterfly plot."""
    from matplotlib import patheffects, pyplot as plt
    from matplotlib.widgets import SpanSelector
    assert len(axes) == len(ch_types_used)
    texts = list()
    idxs = list()
    lines = list()
    path_effects = [patheffects.withStroke(linewidth=2, foreground="w",
                                           alpha=0.75)]
    gfp_path_effects = [patheffects.withStroke(linewidth=5, foreground="w",
                                               alpha=0.75)]
    if selectable:
        selectables = np.ones(len(ch_types_used), dtype=bool)
        for type_idx, this_type in enumerate(ch_types_used):
            idx = picks[types == this_type]
            if len(idx) < 2 or (this_type == 'grad' and len(idx) < 4):
                # prevent unnecessary warnings for e.g. EOG
                if this_type in _DATA_CH_TYPES_SPLIT:
                    logger.info('Need more than one channel to make '
                                'topography for %s. Disabling interactivity.'
                                % (this_type,))
                selectables[type_idx] = False

    if selectable:
        # Parameters for butterfly interactive plots
        params = dict(axes=axes, texts=texts, lines=lines,
                      ch_names=info['ch_names'], idxs=idxs, need_draw=False,
                      path_effects=path_effects)
        fig.canvas.mpl_connect('pick_event',
                               partial(_butterfly_onpick, params=params))
        fig.canvas.mpl_connect('button_press_event',
                               partial(_butterfly_on_button_press,
                                       params=params))
    for ai, (ax, this_type) in enumerate(zip(axes, ch_types_used)):
        line_list = list()  # 'line_list' contains the lines for this axes
        if unit is False:
            this_scaling = 1.0
            ch_unit = 'NA'  # no unit
        else:
            this_scaling = 1. if scalings is None else scalings[this_type]
            ch_unit = units[this_type]
        idx = list(picks[types == this_type])
        idxs.append(idx)

        if len(idx) > 0:
            # Set amplitude scaling
            D = this_scaling * data[idx, :]
            _check_if_nan(D)
            gfp_only = (isinstance(gfp, str) and gfp == 'only')
            if not gfp_only:
                chs = [info['chs'][i] for i in idx]
                locs3d = np.array([ch['loc'][:3] for ch in chs])
                if spatial_colors is True and not _check_ch_locs(chs):
                    warn('Channel locations not available. Disabling spatial '
                         'colors.')
                    spatial_colors = selectable = False
                if spatial_colors is True and len(idx) != 1:
                    x, y, z = locs3d.T
                    colors = _rgb(x, y, z)
                    _handle_spatial_colors(colors, info, idx, this_type, psd,
                                           ax)
                else:
                    if isinstance(spatial_colors, (tuple, str)):
                        col = [spatial_colors]
                    else:
                        col = ['k']
                    colors = col * len(idx)
                    for i in bad_ch_idx:
                        if i in idx:
                            colors[idx.index(i)] = 'r'

                if zorder == 'std':
                    # find the channels with the least activity
                    # to map them in front of the more active ones
                    z_ord = D.std(axis=1).argsort()
                elif zorder == 'unsorted':
                    z_ord = list(range(D.shape[0]))
                elif not callable(zorder):
                    error = ('`zorder` must be a function, "std" '
                             'or "unsorted", not {0}.')
                    raise TypeError(error.format(type(zorder)))
                else:
                    z_ord = zorder(D)

                # plot channels
                for ch_idx, z in enumerate(z_ord):
                    line_list.append(
                        ax.plot(times, D[ch_idx], picker=3.,
                                zorder=z + 1 if spatial_colors is True else 1,
                                color=colors[ch_idx], alpha=line_alpha,
                                linewidth=0.5)[0])

            if gfp:  # 'only' or boolean True
                gfp_color = 3 * (0.,) if spatial_colors is True else (0., 1.,
                                                                      0.)
                this_gfp = np.sqrt((D * D).mean(axis=0))
                this_ylim = ax.get_ylim() if (ylim is None or this_type not in
                                              ylim.keys()) else ylim[this_type]
                if gfp_only:
                    y_offset = 0.
                else:
                    y_offset = this_ylim[0]
                this_gfp += y_offset
                ax.fill_between(times, y_offset, this_gfp, color='none',
                                facecolor=gfp_color, zorder=1, alpha=0.2)
                line_list.append(ax.plot(times, this_gfp, color=gfp_color,
                                         zorder=3, alpha=line_alpha)[0])
                ax.text(times[0] + 0.01 * (times[-1] - times[0]),
                        this_gfp[0] + 0.05 * np.diff(ax.get_ylim())[0],
                        'GFP', zorder=4, color=gfp_color,
                        path_effects=gfp_path_effects)
            for ii, line in zip(idx, line_list):
                if ii in bad_ch_idx:
                    line.set_zorder(2)
                    if spatial_colors is True:
                        line.set_linestyle("--")
            ax.set_ylabel(ch_unit)
            # for old matplotlib, we actually need this to have a bounding
            # box (!), so we have to put some valid text here, change
            # alpha and path effects later
            texts.append(ax.text(0, 0, 'blank', zorder=3,
                                 verticalalignment='baseline',
                                 horizontalalignment='left',
                                 fontweight='bold', alpha=0))

            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and this_type in ylim:
                ax.set_ylim(ylim[this_type])
            ax.set(title=r'%s (%d channel%s)'
                   % (titles[this_type], len(D), _pl(len(D))))
            if ai == 0:
                _add_nave(ax, nave)
            if hline is not None:
                for h in hline:
                    c = ('grey' if spatial_colors is True else 'r')
                    ax.axhline(h, linestyle='--', linewidth=2, color=c)
        lines.append(line_list)
    if selectable:
        for ax in np.array(axes)[selectables]:
            if len(ax.lines) == 1:
                continue
            text = ax.annotate('Loading...', xy=(0.01, 0.1),
                               xycoords='axes fraction', fontsize=20,
                               color='green', zorder=3)
            text.set_visible(False)
            callback_onselect = partial(_line_plot_onselect,
                                        ch_types=ch_types_used, info=info,
                                        data=data, times=times, text=text,
                                        psd=psd, time_unit=time_unit)
            blit = False if plt.get_backend() == 'MacOSX' else True
            minspan = 0 if len(times) < 2 else times[1] - times[0]
            ax._span_selector = SpanSelector(
                ax, callback_onselect, 'horizontal', minspan=minspan,
                useblit=blit, rectprops=dict(alpha=0.5, facecolor='red'))


def _add_nave(ax, nave):
    """Add nave to axes."""
    if nave is not None:
        ax.annotate(
            r'N$_{\mathrm{ave}}$=%d' % nave, ha='left', va='bottom',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(0, 5), textcoords='offset pixels')


def _handle_spatial_colors(colors, info, idx, ch_type, psd, ax):
    """Set up spatial colors."""
    used_nm = np.array(_clean_names(info['ch_names']))[idx]
    # find indices for bads
    bads = [np.where(used_nm == bad)[0][0] for bad in info['bads'] if bad in
            used_nm]
    pos = _auto_topomap_coords(info, idx, ignore_overlap=True, to_sphere=True)
    pos, outlines = _check_outlines(pos, np.array([1, 1]),
                                    {'center': (0, 0), 'scale': (0.5, 0.5)})
    loc = 1 if psd else 2  # Legend in top right for psd plot.
    _plot_legend(pos, colors, ax, bads, outlines, loc)


def _plot_image(data, ax, this_type, picks, cmap, unit, units, scalings, times,
                xlim, ylim, titles, colorbar=True, mask=None, mask_cmap=None,
                mask_style=None, mask_alpha=.25, nave=None,
                time_unit='s', show_names=False, ch_names=None):
    """Plot images."""
    import matplotlib.pyplot as plt
    assert time_unit is not None

    if show_names == "auto":
        if picks is not None:
            show_names = "all" if len(picks) < 25 else True
        else:
            show_names = False

    cmap = _setup_cmap(cmap)

    ch_unit = units[this_type]
    this_scaling = scalings[this_type]
    if unit is False:
        this_scaling = 1.0
        ch_unit = 'NA'  # no unit

    if picks is not None:
        data = data[picks]
        if mask is not None:
            mask = mask[picks]
    # Show the image
    # Set amplitude scaling
    data = this_scaling * data
    if ylim is None or this_type not in ylim:
        vmax = np.abs(data).max()
        vmin = -vmax
    else:
        vmin, vmax = ylim[this_type]

    _check_if_nan(data)

    im, t_end = _plot_masked_image(
        ax, data, times, mask, yvals=None, cmap=cmap[0],
        vmin=vmin, vmax=vmax, mask_style=mask_style, mask_alpha=mask_alpha,
        mask_cmap=mask_cmap)

    if xlim is not None:
        if xlim == 'tight':
            xlim = (times[0], times[-1])
        ax.set_xlim(xlim)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_title(ch_unit)
        if cmap[1]:
            ax.CB = DraggableColorbar(cbar, im)

    ylabel = "Channels" if show_names else 'Channel (index)'
    t = titles[this_type] + ' (%d channel%s' % (len(data), _pl(data)) + t_end
    ax.set(ylabel=ylabel, xlabel='Time (%s)' % (time_unit,), title=t)
    _add_nave(ax, nave)

    if show_names is not False:
        if show_names == "all":
            yticks = np.arange(len(picks)).astype(int)
            yticklabels = np.array(ch_names)[picks]
        else:
            max_tick = len(picks)
            yticks = [tick for tick in ax.get_yticks() if tick < max_tick]
            yticks = np.array(yticks).astype(int)
            # these should only ever be ints right?
            yticklabels = np.array(ch_names)[picks][yticks]
        ax.set(yticks=yticks + .5, yticklabels=yticklabels)


@verbose
def plot_evoked(evoked, picks=None, exclude='bads', unit=True, show=True,
                ylim=None, xlim='tight', proj=False, hline=None, units=None,
                scalings=None, titles=None, axes=None, gfp=False,
                window_title=None, spatial_colors=False, zorder='unsorted',
                selectable=True, noise_cov=None, time_unit='s', verbose=None):
    """Plot evoked data using butterfly plots.

    Left click to a line shows the channel name. Selecting an area by clicking
    and holding left mouse button plots a topographic map of the painted area.

    .. note:: If bad channels are not excluded they are shown in red.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    %(picks_all)s
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
    hline : list of float | None
        The values at which to show an horizontal line.
    units : dict | None
        The units of the channel types used for axes labels. If None,
        defaults to `dict(eeg='uV', grad='fT/cm', mag='fT')`.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,
        defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        `dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')`.
    axes : instance of Axes | list | None
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
    zorder : str | callable
        Which channels to put in the front or back. Only matters if
        `spatial_colors` is used.
        If str, must be `std` or `unsorted` (defaults to `unsorted`). If
        `std`, data with the lowest standard deviation (weakest effects) will
        be put in front so that they are not obscured by those with stronger
        effects. If `unsorted`, channels are z-sorted as in the evoked
        instance.
        If callable, must take one argument: a numpy array of the same
        dimensionality as the evoked raw data; and return a list of
        unique integers corresponding to the number of channels.

        .. versionadded:: 0.13.0

    selectable : bool
        Whether to use interactive features. If True (default), it is possible
        to paint an area to draw topomaps. When False, the interactive features
        are disabled. Disabling interactive features reduces memory consumption
        and is useful when using ``axes`` parameter to draw multiaxes figures.

        .. versionadded:: 0.13.0

    noise_cov : instance of Covariance | str | None
        Noise covariance used to whiten the data while plotting.
        Whitened data channel names are shown in italic.
        Can be a string to load a covariance from disk.
        See also :meth:`mne.Evoked.plot_white` for additional inspection
        of noise covariance properties when whitening evoked data.
        For data processed with SSS, the effective dependence between
        magnetometers and gradiometers may introduce differences in scaling,
        consider using :meth:`mne.Evoked.plot_white`.

        .. versionadded:: 0.16.0
    time_unit : str
        The units for the time axis, can be "ms" or "s" (default).

        .. versionadded:: 0.16
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure containing the butterfly plots.

    See Also
    --------
    mne.viz.plot_evoked_white
    """
    return _plot_evoked(
        evoked=evoked, picks=picks, exclude=exclude, unit=unit, show=show,
        ylim=ylim, proj=proj, xlim=xlim, hline=hline, units=units,
        scalings=scalings, titles=titles, axes=axes, plot_type="butterfly",
        gfp=gfp, window_title=window_title, spatial_colors=spatial_colors,
        selectable=selectable, zorder=zorder, noise_cov=noise_cov,
        time_unit=time_unit)


def plot_evoked_topo(evoked, layout=None, layout_scale=0.945, color=None,
                     border='none', ylim=None, scalings=None, title=None,
                     proj=False, vline=[0.0], fig_background=None,
                     merge_grads=False, legend=True, axes=None,
                     background_color='w', noise_cov=None, show=True):
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
    color : list of color | color | None
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
    vline : list of float | None
        The values at which to show a vertical line.
    fig_background : None | ndarray
        A background image for the figure. This must work with a call to
        plt.imshow. Defaults to None.
    merge_grads : bool
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
    background_color : color
        Background color. Typically 'k' (black) or 'w' (white; default).

        .. versionadded:: 0.15.0
    noise_cov : instance of Covariance | str | None
        Noise covariance used to whiten the data while plotting.
        Whitened data channel names are shown in italic.
        Can be a string to load a covariance from disk.

        .. versionadded:: 0.16.0
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Images of evoked responses at sensor locations
    """
    from matplotlib.colors import colorConverter

    if not type(evoked) in (tuple, list):
        evoked = [evoked]

    dark_background = \
        np.mean(colorConverter.to_rgb(background_color)) < 0.5
    if dark_background:
        fig_facecolor = background_color
        axis_facecolor = background_color
        font_color = 'w'
    else:
        fig_facecolor = background_color
        axis_facecolor = background_color
        font_color = 'k'
    if color is None:
        if dark_background:
            color = ['w'] + _get_color_list()
        else:
            color = _get_color_list()
        color = color * ((len(evoked) % len(color)) + 1)
        color = color[:len(evoked)]
    return _plot_evoked_topo(evoked=evoked, layout=layout,
                             layout_scale=layout_scale, color=color,
                             border=border, ylim=ylim, scalings=scalings,
                             title=title, proj=proj, vline=vline,
                             fig_facecolor=fig_facecolor,
                             fig_background=fig_background,
                             axis_facecolor=axis_facecolor,
                             font_color=font_color, merge_grads=merge_grads,
                             legend=legend, axes=axes, show=show,
                             noise_cov=noise_cov)


@fill_doc
def plot_evoked_image(evoked, picks=None, exclude='bads', unit=True,
                      show=True, clim=None, xlim='tight', proj=False,
                      units=None, scalings=None, titles=None, axes=None,
                      cmap='RdBu_r', colorbar=True, mask=None,
                      mask_style=None, mask_cmap="Greys", mask_alpha=.25,
                      time_unit='s', show_names="auto", group_by=None):
    """Plot evoked data as images.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    %(picks_all)s
        This parameter can also be used to set the order the channels
        are shown in, as the channel image is sorted by the order of picks.
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
        The units of the channel types used for axes labels. If None,
        defaults to ``dict(eeg='uV', grad='fT/cm', mag='fT')``.
    scalings : dict | None
        The scalings of the channel types to be applied for plotting. If None,`
        defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
    titles : dict | None
        The titles associated with the channels. If None, defaults to
        ``dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')``.
    axes : instance of Axes | list | dict | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of channel types. If instance of
        Axes, there must be only one channel type plotted.
        If `group_by` is a dict, this cannot be a list, but it can be a dict
        of lists of axes, with the keys matching those of `group_by`. In that
        case, the provided axes will be used for the corresponding groups.
        Defaults to `None`.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive'
        Colormap. If tuple, the first value indicates the colormap to use and
        the second value is a boolean defining interactivity. In interactive
        mode the colors are adjustable by clicking and dragging the colorbar
        with left and right mouse button. Left mouse button moves the scale up
        and down and right mouse button adjusts the range. Hitting space bar
        resets the scale. Up and down arrows can be used to change the
        colormap. If 'interactive', translates to ``('RdBu_r', True)``.
        Defaults to ``'RdBu_r'``.
    colorbar : bool
        If True, plot a colorbar. Defaults to True.

        .. versionadded:: 0.16
    mask : ndarray | None
        An array of booleans of the same shape as the data. Entries of the
        data that correspond to ```False`` in the mask are masked (see
        `do_mask` below). Useful for, e.g., masking for statistical
        significance.

        .. versionadded:: 0.16
    mask_style: None | 'both' | 'contour' | 'mask'
        If `mask` is not None: if 'contour', a contour line is drawn around
        the masked areas (``True`` in `mask`). If 'mask', entries not
        ``True`` in `mask` are shown transparently. If 'both', both a contour
        and transparency are used.
        If ``None``, defaults to 'both' if `mask` is not None, and is ignored
        otherwise.

         .. versionadded:: 0.16
    mask_cmap : matplotlib colormap | (colormap, bool) | 'interactive'
        The colormap chosen for masked parts of the image (see below), if
        `mask` is not ``None``. If None, `cmap` is reused. Defaults to
        ``Greys``. Not interactive. Otherwise, as `cmap`.
    mask_alpha : float
        A float between 0 and 1. If `mask` is not None, this sets the
        alpha level (degree of transparency) for the masked-out segments.
        I.e., if 0, masked-out segments are not visible at all.
        Defaults to .25.

        .. versionadded:: 0.16
    time_unit : str
        The units for the time axis, can be "ms" or "s" (default).

        .. versionadded:: 0.16
    show_names : bool | str
        Determines if channel names should be plotted on the y axis. If False,
        no names are shown. If True, ticks are set automatically and the
        corresponding channel names are shown. If str, must be "auto" or "all".
        If "all", all channel names are shown.
        If "auto", is set to False if `picks` is ``None``; to ``True`` if
        `picks` is not ``None`` and fewer than 25 picks are shown; to "all"
        if `picks` is not ``None`` and contains fewer than 25 entries.
    group_by : None | dict
        If a dict, the values must be picks, and `axes` must also be a dict
        with matching keys, or None. If `axes` is None, one figure and one axis
        will be created for each entry in `group_by`.
        Then, for each entry, the picked channels will be plotted
        to the corresponding axis. If `titles` are None, keys will become plot
        titles. This is useful for e.g. ROIs. Each entry must contain only
        one channel type. For example::

            group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

        If None, all picked channels are plotted to the same axis.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure containing the images.
    """
    return _plot_evoked(evoked=evoked, picks=picks, exclude=exclude, unit=unit,
                        show=show, ylim=clim, proj=proj, xlim=xlim, hline=None,
                        units=units, scalings=scalings, titles=titles,
                        axes=axes, plot_type="image", cmap=cmap,
                        colorbar=colorbar, mask=mask, mask_style=mask_style,
                        mask_cmap=mask_cmap, mask_alpha=mask_alpha,
                        time_unit=time_unit, show_names=show_names,
                        group_by=group_by)


def _plot_update_evoked(params, bools):
    """Update the plot evoked lines."""
    picks, evoked = [params[k] for k in ('picks', 'evoked')]
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
                line.set_ydata(di)
        else:
            ax.images[0].set_data(D)
    params['fig'].canvas.draw()


@verbose
def plot_evoked_white(evoked, noise_cov, show=True, rank=None, time_unit='s',
                      verbose=None):
    u"""Plot whitened evoked response.

    Plots the whitened evoked response and the whitened GFP as described in
    [1]_. This function is especially useful for investigating noise
    covariance properties to determine if data are properly whitened (e.g.,
    achieving expected values in line with model assumptions, see Notes below).

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The evoked response.
    noise_cov : list | instance of Covariance | str
        The noise covariance. Can be a string to load a covariance from disk.
    show : bool
        Show figure if True.
    %(rank_None)s
    time_unit : str
        The units for the time axis, can be "ms" or "s" (default).

        .. versionadded:: 0.16
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.

    See Also
    --------
    mne.Evoked.plot

    Notes
    -----
    If baseline signals match the assumption of Gaussian white noise,
    values should be centered at 0, and be within 2 standard deviations
    (±1.96) for 95%% of the time points. For the global field power (GFP),
    we expect it to fluctuate around a value of 1.

    If one single covariance object is passed, the GFP panel (bottom)
    will depict different sensor types. If multiple covariance objects are
    passed as a list, the left column will display the whitened evoked
    responses for each channel based on the whitener from the noise covariance
    that has the highest log-likelihood. The left column will depict the
    whitened GFPs based on each estimator separately for each sensor type.
    Instead of numbers of channels the GFP display shows the estimated rank.
    Note. The rank estimation will be printed by the logger
    (if ``verbose=True``) for each noise covariance estimator that is passed.

    References
    ----------
    .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
           covariance estimation and spatial whitening of MEG and EEG
           signals, vol. 108, 328-342, NeuroImage.
    """
    return _plot_evoked_white(evoked=evoked, noise_cov=noise_cov,
                              scalings=None, rank=rank, show=show,
                              time_unit=time_unit)


def _plot_evoked_white(evoked, noise_cov, scalings=None, rank=None, show=True,
                       time_unit='s'):
    """Help plot_evoked_white.

    Additional Parameters
    ---------------------
    scalings : dict | None
        The rescaling method to be applied to improve the accuracy of rank
        estimaiton. If dict, it will override the following default values
        (used if None)::

            dict(mag=1e12, grad=1e11, eeg=1e5)

        Note. These values were tested on different datests across various
        conditions. You should not need to update them.

    """
    from ..cov import whiten_evoked, read_cov  # recursive import
    import matplotlib.pyplot as plt
    time_unit, times = _check_time_unit(time_unit, evoked.times)

    if isinstance(noise_cov, str):
        noise_cov = read_cov(noise_cov)
    if not isinstance(noise_cov, (list, tuple)):
        noise_cov = [noise_cov]

    evoked = evoked.copy()  # handle ref meg
    passive_idx = [idx for idx, proj in enumerate(evoked.info['projs'])
                   if not proj['active']]
    # either applied already or not-- else issue
    for idx in passive_idx[::-1]:  # reverse order so idx does not change
        evoked.del_proj(idx)

    evoked.pick_types(ref_meg=False, exclude='bads', **_PICK_TYPES_DATA_DICT)
    n_ch_used, rank_list, picks_list, has_sss = _triage_rank_sss(
        evoked.info, noise_cov, rank, scalings)
    del rank, scalings
    if has_sss:
        logger.info('SSS has been applied to data. Showing mag and grad '
                    'whitening jointly.')

    # get one whitened evoked per cov
    evokeds_white = [whiten_evoked(evoked, cov, picks=None, rank=r)
                     for cov, r in zip(noise_cov, rank_list)]

    def whitened_gfp(x, rank=None):
        """Whitened Global Field Power.

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

    if any(((n_columns == 1 and n_ch_used >= 1),
            (n_columns == 2 and n_ch_used == 1))):
        axes_evoked = axes[:n_ch_used]
        ax_gfp = axes[-1:]
    elif n_columns == 2 and n_ch_used > 1:
        axes_evoked = axes[:n_ch_used, 0]
        ax_gfp = axes[:, 1]
    else:
        raise RuntimeError('Wrong axes inputs')

    titles_ = _handle_default('titles')
    if has_sss:
        titles_['meg'] = 'MEG (combined)'

    colors = [plt.cm.Set1(i) for i in np.linspace(0, 0.5, len(noise_cov))]
    ch_colors = _handle_default('color', None)
    iter_gfp = zip(evokeds_white, noise_cov, rank_list, colors)

    # the first is by law the best noise cov, on the left we plot that one.
    if not has_sss:
        evokeds_white[0].plot(unit=False, axes=axes_evoked,
                              hline=[-1.96, 1.96], show=False,
                              time_unit=time_unit)
    else:
        for ((ch_type, picks), ax) in zip(picks_list, axes_evoked):
            ax.plot(times, evokeds_white[0].data[picks].T, color='k',
                    lw=0.5)
            for hline in [-1.96, 1.96]:
                ax.axhline(hline, color='red', linestyle='--', lw=2)
            ax.set(title='%s (%d channel%s)'
                   % (titles_[ch_type], len(picks), _pl(len(picks))))

    # Now plot the GFP for all covs if indicated.
    for evoked_white, noise_cov, rank_, color in iter_gfp:
        i = 0

        for ch, sub_picks in picks_list:
            this_rank = rank_[ch]
            title = '{0} ({2}{1})'.format(
                    titles_[ch] if n_columns > 1 else ch,
                    this_rank, 'rank ' if n_columns > 1 else '')
            label = noise_cov.get('method', 'empirical')

            ax = ax_gfp[i]
            ax.set_title(title if n_columns > 1 else
                         'Whitened GFP, method = "%s"' % label)

            data = evoked_white.data[sub_picks]
            gfp = whitened_gfp(data, rank=this_rank)
            # Wrap SSS-processed data (MEG) to the mag color
            color_ch = 'mag' if ch == 'meg' else ch
            ax.plot(times, gfp,
                    label=label if n_columns > 1 else title,
                    color=color if n_columns > 1 else ch_colors[color_ch],
                    lw=0.5)
            ax.set(xlabel='Time (%s)' % (time_unit,), ylabel=r'GFP ($\chi^2$)',
                   xlim=[times[0], times[-1]], ylim=(0, 10))
            ax.axhline(1, color='red', linestyle='--', lw=2.)
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


@verbose
def plot_snr_estimate(evoked, inv, show=True, verbose=None):
    """Plot a data SNR estimate.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance. This should probably be baseline-corrected.
    inv : instance of InverseOperator
        The minimum-norm inverse operator.
    show : bool
        Show figure if True.
    %(verbose)s

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    The bluish green line is the SNR determined by the GFP of the whitened
    evoked data. The orange line is the SNR estimated based on the mismatch
    between the data and the data re-estimated from the regularized inverse.

    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    from ..minimum_norm import estimate_snr
    snr, snr_est = estimate_snr(evoked, inv)
    fig, ax = plt.subplots(1, 1)
    lims = np.concatenate([evoked.times[[0, -1]], [-1, snr_est.max()]])
    ax.axvline(0, color='k', ls=':', lw=1)
    ax.axhline(0, color='k', ls=':', lw=1)
    # Colors are "bluish green" and "vermilion" taken from:
    #  http://bconnelly.net/2013/10/creating-colorblind-friendly-figures/
    ax.plot(evoked.times, snr_est, color=[0.0, 0.6, 0.5])
    ax.plot(evoked.times, snr - 1, color=[0.8, 0.4, 0.0])
    ax.set(xlim=lims[:2], ylim=lims[2:], ylabel='SNR', xlabel='Time (s)')
    if evoked.comment is not None:
        ax.set_title(evoked.comment)
    plt_show(show)
    return fig


@fill_doc
def plot_evoked_joint(evoked, times="peaks", title='', picks=None,
                      exclude=None, show=True, ts_args=None,
                      topomap_args=None):
    """Plot evoked data as butterfly plot and add topomaps for time points.

    .. note:: Axes to plot in can be passed by the user through ``ts_args`` or
              ``topomap_args``. In that case both ``ts_args`` and
              ``topomap_args`` axes have to be used. Be aware that when the
              axes are provided, their position may be slightly modified.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance.
    times : float | array of float | "auto" | "peaks"
        The time point(s) to plot. If ``"auto"``, 5 evenly spaced topographies
        between the first and last time instant will be shown. If ``"peaks"``,
        finds time points automatically by checking for 3 local maxima in
        Global Field Power. Defaults to ``"peaks"``.
    title : str | None
        The title. If ``None``, suppress printing channel type title. If an
        empty string, a default title is created. Defaults to ''. If custom
        axes are passed make sure to set ``title=None``, otherwise some of your
        axes may be removed during placement of the title axis.
    %(picks_all)s
    exclude : None | list of str | 'bads'
        Channels names to exclude from being shown. If ``'bads'``, the
        bad channels are excluded. Defaults to ``None``.
    show : bool
        Show figure if ``True``. Defaults to ``True``.
    ts_args : None | dict
        A dict of ``kwargs`` that are forwarded to :meth:`mne.Evoked.plot` to
        style the butterfly plot. If they are not in this dict, the following
        defaults are passed: ``spatial_colors=True``, ``zorder='std'``.
        ``show`` and ``exclude`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.
    topomap_args : None | dict
        A dict of `kwargs` that are forwarded to
        :meth:`mne.Evoked.plot_topomap` to style the topomaps.
        If it is not in this dict, ``outlines='skirt'`` will be passed.
        ``show``, ``times``, ``colorbar`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure | list
        The figure object containing the plot. If ``evoked`` has multiple
        channel types, a list of figures, one for each channel type, is
        returned.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    import matplotlib.pyplot as plt

    if ts_args is not None and not isinstance(ts_args, dict):
        raise TypeError('ts_args must be dict or None, got type %s'
                        % (type(ts_args),))
    ts_args = dict() if ts_args is None else ts_args.copy()
    ts_args['time_unit'], _ = _check_time_unit(
        ts_args.get('time_unit', 's'), evoked.times)
    if topomap_args is None:
        topomap_args = dict()

    got_axes = False
    illegal_args = {"show", 'times', 'exclude'}
    for args in (ts_args, topomap_args):
        if any((x in args for x in illegal_args)):
            raise ValueError("Don't pass any of {} as *_args.".format(
                ", ".join(list(illegal_args))))
    if ("axes" in ts_args) or ("axes" in topomap_args):
        if not (("axes" in ts_args) and ("axes" in topomap_args)):
            raise ValueError("If one of `ts_args` and `topomap_args` contains "
                             "'axes', the other must, too.")
        _validate_if_list_of_axes([ts_args["axes"]], 1)
        n_topomaps = (3 if times is None else len(times)) + 1
        _validate_if_list_of_axes(list(topomap_args["axes"]), n_topomaps)
        got_axes = True

    # channel selection
    # simply create a new evoked object with the desired channel selection
    evoked = _pick_inst(evoked, picks, exclude, copy=True)
    info = evoked.info
    ch_types = _get_channel_types(info, restrict_data_types=True)

    # if multiple sensor types: one plot per channel type, recursive call
    if len(ch_types) > 1:
        if got_axes:
            raise NotImplementedError(
                "Currently, passing axes manually (via `ts_args` or "
                "`topomap_args`) is not supported for multiple channel types.")
        figs = list()
        for this_type in ch_types:  # pick only the corresponding channel type
            ev_ = evoked.copy().pick_channels(
                [info['ch_names'][idx] for idx in range(info['nchan'])
                 if channel_type(info, idx) == this_type])
            if len(_get_channel_types(ev_.info)) > 1:
                raise RuntimeError('Possibly infinite loop due to channel '
                                   'selection problem. This should never '
                                   'happen! Please check your channel types.')
            figs.append(
                plot_evoked_joint(
                    ev_, times=times, title=title, show=show, ts_args=ts_args,
                    exclude=list(), topomap_args=topomap_args))
        return figs

    # set up time points to show topomaps for
    times_sec = _process_times(evoked, times, few=True)
    del times
    _, times_ts = _check_time_unit(ts_args['time_unit'], times_sec)

    # prepare axes for topomap
    if not got_axes:
        fig, ts_ax, map_ax, cbar_ax = _prepare_joint_axes(len(times_sec),
                                                          figsize=(8.0, 4.2))
    else:
        ts_ax = ts_args["axes"]
        del ts_args["axes"]
        map_ax = topomap_args["axes"][:-1]
        cbar_ax = topomap_args["axes"][-1]
        del topomap_args["axes"]
        fig = cbar_ax.figure

    # butterfly/time series plot
    # most of this code is about passing defaults on demand
    ts_args_def = dict(picks=None, unit=True, ylim=None, xlim='tight',
                       proj=False, hline=None, units=None, scalings=None,
                       titles=None, gfp=False, window_title=None,
                       spatial_colors=True, zorder='std')
    ts_args_def.update(ts_args)
    _plot_evoked(evoked, axes=ts_ax, show=False, plot_type='butterfly',
                 exclude=[], set_tight_layout=False, **ts_args_def)

    # handle title
    # we use a new axis for the title to handle scaling of plots
    old_title = ts_ax.get_title()
    ts_ax.set_title('')
    if title is not None:
        title_ax = plt.subplot(4, 3, 2)
        if title == '':
            title = old_title
        title_ax.text(.5, .5, title, transform=title_ax.transAxes,
                      horizontalalignment='center',
                      verticalalignment='center')
        title_ax.axis('off')

    # topomap
    contours = topomap_args.get('contours', 6)
    ch_type = ch_types.pop()  # set should only contain one element
    # Since the data has all the ch_types, we get the limits from the plot.
    vmin, vmax = ts_ax.get_ylim()
    norm = ch_type == 'grad'
    vmin = 0 if norm else vmin
    vmin, vmax = _setup_vmin_vmax(evoked.data, vmin, vmax, norm)
    if not isinstance(contours, (list, np.ndarray)):
        locator, contours = _set_contour_locator(vmin, vmax, contours)
    else:
        locator = None

    topomap_args_pass = topomap_args.copy()
    topomap_args_pass['outlines'] = topomap_args.get('outlines', 'skirt')
    topomap_args_pass['contours'] = contours
    evoked.plot_topomap(times=times_sec, axes=map_ax, show=False,
                        colorbar=False, **topomap_args_pass)

    if topomap_args.get('colorbar', True):
        from matplotlib import ticker
        cbar = plt.colorbar(map_ax[0].images[0], cax=cbar_ax)
        if isinstance(contours, (list, np.ndarray)):
            cbar.set_ticks(contours)
        else:
            if locator is None:
                locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = locator
        cbar.update_ticks()

    if not got_axes:
        plt.subplots_adjust(left=.1, right=.93, bottom=.14,
                            top=1. if title is not None else 1.2)

    # connection lines
    # draw the connection lines between time series and topoplots
    lines = [_connection_line(timepoint, fig, ts_ax, map_ax_)
             for timepoint, map_ax_ in zip(times_ts, map_ax)]
    for line in lines:
        fig.lines.append(line)

    # mark times in time series plot
    for timepoint in times_ts:
        ts_ax.axvline(timepoint, color='grey', linestyle='-',
                      linewidth=1.5, alpha=.66, zorder=0)

    # show and return it
    plt_show(show)
    return fig


###############################################################################
# The following functions are all helpers for plot_compare_evokeds.           #
###############################################################################

def _aux_setup_styles(conditions, style_dict, style, default):
    """Set linestyles and colors for plot_compare_evokeds."""
    # check user-supplied style to condition matching
    tags = {tag for cond in conditions for tag in cond.split("/")}
    msg = ("Can't map between conditions and the provided {0}. Make sure "
           "you have provided keys in the format of '/'-separated tags, "
           "and that these correspond to '/'-separated tags for the condition "
           "names (e.g., conditions like 'Visual/Right', and styles like "
           "'colors=dict(Visual='red'))'. The offending tag was '{1}'.")
    for key in style_dict:
        for tag in key.split("/"):
            if tag not in tags:
                raise ValueError(msg.format(style, tag))

    style_dict = deepcopy(style_dict)

    # check condition to style matching, and fill in defaults
    condition_warning = "Condition {0} could not be mapped to a " + style
    style_warning = ". Using the default of {}.".format(default)
    for condition in conditions:
        if condition not in style_dict:
            if "/" not in condition:
                warn(condition_warning.format(condition) + style_warning)
                style_dict[condition] = default
            for style_ in style_dict:
                if style_ in condition.split("/"):
                    style_dict[condition] = style_dict[style_]
                    break

    return style_dict


def _truncate_yaxis(axes, ymin, ymax, orig_ymin, orig_ymax, fraction,
                    any_positive, any_negative, truncation_style):
    """Truncate the y axis in plot_compare_evokeds."""
    if truncation_style != "max_ticks":
        abs_lims = (orig_ymax if orig_ymax > np.abs(orig_ymin)
                    else np.abs(orig_ymin))
        ymin_, ymax_ = (-(abs_lims // fraction), abs_lims // fraction)
        # user supplied ymin and ymax overwrite everything
        if ymin is not None and ymin > ymin_:
            ymin_ = ymin
        if ymax is not None and ymax < ymax_:
            ymax_ = ymax
        yticks = (ymin_ if any_negative else 0, ymax_ if any_positive else 0)
        axes.set_yticks(yticks)
        ymin_bound, ymax_bound = (-(abs_lims // fraction),
                                  abs_lims // fraction)
        # user supplied ymin and ymax still overwrite everything
        if ymin is not None and ymin > ymin_bound:
            ymin_bound = ymin
        if ymax is not None and ymax < ymax_bound:
            ymax_bound = ymax
        precision = 0.25  # round to .25
        if ymin is None:
            ymin_bound = round(ymin_bound / precision) * precision
        if ymin is None:
            ymax_bound = round(ymax_bound / precision) * precision
        axes.spines['left'].set_bounds(ymin_bound, ymax_bound)
    else:  # code stolen from seaborn
        yticks = axes.get_yticks()
        firsttick = np.compress(yticks >= min(axes.get_ylim()),
                                yticks)[0]
        lasttick = np.compress(yticks <= max(axes.get_ylim()),
                               yticks)[-1]
        axes.spines['left'].set_bounds(firsttick, lasttick)
        newticks = yticks.compress(yticks <= lasttick)
        newticks = newticks.compress(newticks >= firsttick)
        axes.set_yticks(newticks)
        ymin_bound, ymax_bound = newticks[[0, -1]]
    return ymin_bound, ymax_bound


def _combine_grad(evoked, picks):
    """Create a new instance of Evoked with combined gradiometers (RMSE)."""
    def pair_and_combine(data):
        data = data ** 2
        data = (data[::2, :] + data[1::2, :]) / 2
        return np.sqrt(data)
    picks, ch_names = _grad_pair_pick_and_name(evoked.info, picks)
    this_data = pair_and_combine(evoked.data[picks, :])
    ch_names = ch_names[::2]
    evoked = evoked.copy().pick_channels(ch_names)
    combined_ch_names = [ch_name[:-1] + "X" for ch_name in ch_names]
    evoked.rename_channels({c_old: c_new for c_old, c_new
                            in zip(evoked.ch_names, combined_ch_names)})
    evoked.data = this_data
    return evoked


def _check_loc_legal(loc, what='your choice', default=1):
    """Check if loc is a legal location for MPL subordinate axes."""
    true_default = {"show_legend": 3, "show_sensors": 4}.get(what, default)
    if isinstance(loc, (bool, np.bool_)) and loc:
        loc = true_default
    loc_dict = {'upper right': 1, 'upper left': 2, 'lower left': 3,
                'lower right': 4, 'right': 5, 'center left': 6,
                'center right': 7, 'lower center': 8, 'upper center': 9,
                'center': 10}
    loc_ = loc_dict.get(loc, loc)
    if loc_ not in range(11):
        raise ValueError(str(loc) + " is not a legal MPL loc, please supply"
                         "another value for " + what + ".")
    return loc_


def _format_evokeds_colors(evokeds, cmap, colors):
    """Set up to have evokeds as a dict as well as colors."""
    from ..evoked import Evoked, _check_evokeds_ch_names_times

    if isinstance(evokeds, Evoked):
        evokeds = dict(Evoked=evokeds)  # title becomes 'Evoked'
    elif not isinstance(evokeds, dict):  # it's assumed to be a list
        if (cmap is not None) and (colors is None):
            colors = {str(ii + 1): ii for ii, _ in enumerate(evokeds)}
        evokeds = {str(ii + 1): evoked for ii, evoked in enumerate(evokeds)}
    else:
        assert isinstance(evokeds, dict)
        if (colors is None) and cmap is not None:
            raise ValueError('If evokeds is a dict and a cmap is passed, '
                             'you must specify the colors.')
    for cond in evokeds.keys():
        _validate_type(cond, 'str', "Conditions")
    # Now make sure all values are list of Evoked objects
    evokeds = {condition: [v] if isinstance(v, Evoked) else v
               for condition, v in evokeds.items()}

    # Check that all elements are of type evoked
    for this_evoked in evokeds.values():
        for ev in this_evoked:
            _validate_type(ev, Evoked, "All evokeds entries ", "Evoked")

    # Check that all evoked objects have the same time axis and channels
    all_evoked = sum(evokeds.values(), [])
    _check_evokeds_ch_names_times(all_evoked)

    return evokeds, colors


def _setup_styles(conditions, styles, cmap, colors, linestyles):
    """Set up plotting styles for each condition."""
    import matplotlib.pyplot as plt
    # continuous colors
    the_colors, color_conds, color_order = None, None, None
    colors_are_float = False
    if cmap is not None:
        for color_value in colors.values():
            try:
                float(color_value)
            except ValueError:
                raise TypeError("If ``cmap`` is not None, the values of "
                                "``colors`` must be numeric. Got %s" %
                                type(color_value))
        cmapper = getattr(plt.cm, cmap, cmap)
        color_conds = list(colors.keys())
        all_colors = [colors[cond] for cond in color_conds]
        color_order = np.array(all_colors).argsort()
        color_indices = color_order.argsort()

        if all([isinstance(color, Integral) for color in all_colors]):
            msg = "Integer colors detected, mapping to rank positions ..."
            n_colors = len(all_colors)
            colors_ = {cond: ind for cond, ind in
                       zip(color_conds, color_indices)}

            def convert_colors(color):
                return colors_[color]
        else:
            for color in all_colors:
                if not 0 <= color <= 1:
                    raise ValueError("Values of colors must be all-integer or "
                                     "floats between 0 and 1, got %s." % color)
            msg = "Float colors detected, mapping to percentiles ..."
            n_colors = 101  # percentiles plus 1 if we have 1.0s
            colors_old = colors.copy()

            def convert_colors(color):
                return int(colors_old[color] * 100)
            colors_are_float = True
        logger.info(msg)
        the_colors = cmapper(np.linspace(0, 1, n_colors))

        colors = dict()
        for cond in conditions:
            cond_ = cond.split("/")
            for color in color_conds:
                if color in cond_:
                    colors[cond] = the_colors[convert_colors(color)]
                    continue

    # categorical colors
    if not isinstance(colors, dict):
        colors_ = _get_color_list()
        if len(conditions) > len(colors_):
            msg = ("Trying to plot more than {0} conditions. We provide"
                   "only {0} default colors. Please supply colors manually.")
            raise ValueError(msg.format(len(colors_)))
        colors = {condition: color for condition, color
                  in zip(conditions, colors_)}
    else:
        colors = _aux_setup_styles(conditions, colors, "color", "grey")

    # linestyles
    if not isinstance(linestyles, dict):
        linestyles = {condition: linestyle for condition, linestyle in
                      zip(conditions, ['-'] * len(conditions))}
    else:
        linestyles = _aux_setup_styles(conditions, linestyles,
                                       "linestyle", "-")

    # finally, put it all together
    styles = dict() if styles is None else deepcopy(styles)

    for condition, color, linestyle in zip(conditions, colors, linestyles):
        styles[condition] = styles.get(condition, dict())
        styles[condition]['c'] = styles[condition].get('c', colors[condition])
        styles[condition]['linestyle'] = styles[condition].get(
            'linestyle', linestyles[condition])

    return styles, the_colors, color_conds, color_order, colors_are_float


def _evoked_sensor_legend(info, picks, ymin, ymax, show_sensors, ax):
    """Show sensor legend (location of a set of sensors on the head)."""
    if show_sensors is True:
        ymin, ymax = np.abs(ax.get_ylim())
        show_sensors = "lower right" if ymin > ymax else "upper right"

    pos = _auto_topomap_coords(info, picks, ignore_overlap=True,
                               to_sphere=True)
    head_pos = {'center': (0, 0), 'scale': (0.5, 0.5)}
    pos, outlines = _check_outlines(pos, np.array([1, 1]), head_pos)

    show_sensors = _check_loc_legal(show_sensors, "show_sensors")
    _plot_legend(pos, ["k"] * len(picks), ax, list(), outlines,
                 show_sensors, size=25)


def _evoked_condition_legend(conditions, show_legend, split_legend, cmap,
                             colors_are_float, the_colors, colors, color_conds,
                             color_order, cmap_label, linestyles, ax, do_topo):
    """Show condition legend for line plot. Helper for plot_compare_evokeds."""
    import matplotlib.lines as mlines

    if split_legend is None:
        split_legend = cmap is not None  # default to True iff cmap is given
    if split_legend is True:
        if colors is None:
            raise ValueError(
                "If `split_legend` is True, `colors` must not be None.")
        # mpl 1.3 requires us to split it like this. with recent mpl,
        # we could use the label parameter of the Line2D
        legend_lines, legend_labels = list(), list()
        if cmap is None:  # ... one set of lines for the colors
            for color in sorted(colors.keys()):
                line = mlines.Line2D([], [], linestyle="-",
                                     color=colors[color])
                legend_lines.append(line)
                legend_labels.append(color)
        if len(list(linestyles)) > 1:  # ... one set for the linestyle
            for style, s in linestyles.items():
                line = mlines.Line2D([], [], color='k', linestyle=s)
                legend_lines.append(line)
                legend_labels.append(style)

    # the condition legend
    if len(conditions) > 1 and not (
            isinstance(show_legend, (bool, np.bool_)) and not show_legend):
        show_legend_orig = show_legend
        show_legend = _check_loc_legal(show_legend, "show_legend")
        legend_params = dict(loc=show_legend, frameon=True)

        # override if topoplot and default loc
        if do_topo and (
                isinstance(show_legend_orig, bool) and show_legend_orig):
            legend_params["loc"] = "lower right"
            legend_params["bbox_to_anchor"] = (1, 1)
        if split_legend:
            if len(legend_lines) > 1:
                ax.legend(
                    legend_lines, legend_labels,  # see above: mpl 1.3
                    ncol=1 + (len(legend_lines) // 4), **legend_params)
        else:
            ax.legend(ncol=1 + (len(conditions) // 5), **legend_params)

    # the colormap, if `cmap` is provided
    if split_legend and cmap is not None:
        # plot the colorbar ... complicated cause we don't have a heatmap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        if colors_are_float:
            ax_cb.imshow(the_colors[:, np.newaxis, :], interpolation='none',
                         aspect=.05)
            color_ticks = np.array(list(set(colors.values()))) * 100
            ax_cb.set_yticks(color_ticks)
            ax_cb.set_yticklabels(color_ticks)
        else:
            ax_cb.imshow(the_colors[:, np.newaxis, :], interpolation='none')
            ax_cb.set_yticks(np.arange(len(the_colors)))
            ax_cb.set_yticklabels(np.array(color_conds)[color_order])
        ax_cb.yaxis.tick_right()
        ax_cb.set(xticks=(), ylabel=cmap_label)

    if do_topo:
        # we need the lines for the legends, but then we can kill them
        ax.lines.clear()
        ax.set_title("")
        del ax.texts[-1]


def _set_ylims_plot_compare_evokeds(ax, any_positive, any_negative, ymin, ymax,
                                    truncate_yaxis,  truncate_xaxis, invert_y,
                                    vlines, tmin, tmax, unit,
                                    skip_axlabel=True):
    """Set ylims for an evoked plot. Helper for plot_compare_evokeds."""
    # truncate the y axis - this is aesthetics
    orig_ymin, orig_ymax = ax.get_ylim()
    if not any_positive:
        orig_ymax = 0
    if not any_negative:
        orig_ymin = 0

    ax.set_ylim(orig_ymin if ymin is None else ymin,
                orig_ymax if ymax is None else ymax)

    fraction = 2 if ax.get_ylim()[0] >= 0 else 3

    if truncate_yaxis:
        _, ymax_bound = _truncate_yaxis(
            ax, ymin, ymax, orig_ymin, orig_ymax, fraction,
            any_positive, any_negative, truncate_yaxis)
    else:
        if ymin is not None and ymin > 0:
            warn("ymin is all-positive, not truncating yaxis")
        ymax_bound = ax.get_ylim()[-1]

    current_ymin = ax.get_ylim()[0]

    # plot v lines
    # Why 'invert_y'? Many EEG people plot negative values up for ... reasons
    if invert_y and (current_ymin < 0):
        upper_v, lower_v = -ymax_bound, ax.get_ylim()[-1]
    else:
        upper_v, lower_v = ax.get_ylim()[0], ymax_bound
    if vlines:
        ax.vlines(vlines, upper_v, lower_v, linestyles='--', colors='k',
                  linewidth=1., zorder=1)

    # more aesthetics
    _setup_ax_spines(ax, vlines, tmin, tmax, invert_y, ymax_bound, unit,
                     truncate_xaxis, skip_axlabel=skip_axlabel)


def _get_data_and_ci(evoked, scaling=1, picks=None, ci_fun=None, gfp=False):
    """Compute (sensor-aggregated, scaled) time series and possibly CI."""
    from .. import Evoked
    if picks is None:
        picks = Ellipsis
    picks = np.array(picks).flatten()
    if not isinstance(evoked, Evoked):  # ... it is a list of evokeds
        data = np.array([e.data[picks, :] * scaling for e in evoked])
    else:
        data = evoked.data[np.newaxis, picks] * scaling

    if gfp:
        data = np.sqrt(np.mean(data * data, axis=1))
    else:
        data = np.mean(data, axis=1)  # average across channels

    if ci_fun is not None:  # compute CI if requested:
        ci = ci_fun(data)

    # average across trials:
    data = np.mean(data, axis=0)
    _check_if_nan(data)

    if ci_fun is not None:
        return data, ci
    else:
        return data,


def _calculate_ci_and_mean(evokeds, conditions, scaling, picks, ci_fun, gfp):
    """Calculate time series and CI, potentially aggregating over sensors."""
    ci_dict, data_dict = dict(), dict()

    for cond in conditions:
        this_evokeds = evokeds[cond]
        # this will fail if evokeds do not have the same structure
        # (e.g. channel count)
        res = _get_data_and_ci(this_evokeds, scaling=scaling, picks=picks,
                               ci_fun=ci_fun, gfp=gfp)
        data_dict[cond] = res[0]
        if ci_fun is not None:
            ci_dict[cond] = res[1]
    return data_dict, ci_dict


def _get_ci_function_for_evokeds(ci):
    """Get the function for calculating the confidence interval for evokeds."""
    # check ci parameter
    if ci is None:
        return None

    if ci is True:
        ci = .95
    elif ci is not False and not (isinstance(ci, np.float) or callable(ci)):
        raise TypeError('ci must be None, bool, float or callable, got %s' %
                        type(ci))

    _ci_fun = None
    if ci is not False:
        if callable(ci):
            _ci_fun = ci
        else:
            from ..stats import _ci
            _ci_fun = partial(_ci, ci=ci, method="bootstrap")
    return _ci_fun


def _finish_styles_plot_comp_evoked(styles, colors, linestyles, conditions,
                                    cmap):
    """Finalize styles for plot_compare_evokeds."""
    # Styles (especially color and linestyle) are pulled from a dict 'styles'.
    # This dict has one entry per condition. Its color and linestyle entries
    # are pulled from the 'colors' and 'linestyles' dicts via '/'-tag matching
    # unless they are overwritten by entries from a user-provided 'styles'.

    # check if input is valid
    if isinstance(styles, dict):
        for style_ in styles:
            if style_ not in conditions:
                raise ValueError("Could not map between 'styles' and "
                                 "conditions. Condition " + style_ +
                                 " was not found in the supplied data.")

    # color
    if (colors is not None and not isinstance(colors, str) and
            not isinstance(colors, dict) and len(colors) > 1):
        colors = {condition: color for condition, color
                  in zip(conditions, colors)}

    cmap_label = ""
    if cmap is not None:
        if not isinstance(cmap, str) and len(cmap) == 2:
            cmap_label, cmap = cmap

    styles, the_colors, color_conds, color_order, colors_are_float =\
        _setup_styles(conditions, styles, cmap, colors, linestyles)

    return (styles, colors, the_colors, color_conds, color_order,
            colors_are_float, cmap_label, cmap)


def _plot_compare_evokeds(ax, data_dict, conditions, times, do_ci, ci_dict,
                          styles, title, all_positive, topo):
    """Plot evokeds (to compare them; with CIs) based on a data_dict."""
    any_negative, any_positive = False, False
    for condition in conditions:
        # plot the actual data ('d') as a line
        d = data_dict[condition].T
        ax.plot(times, d, zorder=1000, label=condition, clip_on=False,
                **styles[condition])
        if any_positive or np.any(d > 0):
            any_positive = True
        if any_negative or np.any(d < 0):
            any_negative = True

        # plot the confidence interval if available
        if do_ci:
            ci_ = ci_dict[condition]
            ax.fill_between(times, ci_[0].flatten(), ci_[1].flatten(),
                            zorder=9, color=styles[condition]['c'], alpha=.3,
                            clip_on=False)
    if topo:
        ax.text(-.1, 1, title, transform=ax.transAxes)
    else:
        ax.set_title(title)

    return any_positive, any_negative


@fill_doc
def plot_compare_evokeds(evokeds, picks=None, gfp=False, colors=None,
                         linestyles=['-'], styles=None, cmap=None,
                         vlines="auto", ci=0.95, truncate_yaxis="max_ticks",
                         truncate_xaxis=True, ylim=None, invert_y=False,
                         show_sensors=None, show_legend=True,
                         split_legend=False, axes=None, title=None, show=True):
    """Plot evoked time courses for one or more conditions and/or channels.

    Parameters
    ----------
    evokeds : instance of mne.Evoked | list | dict
        If a single Evoked instance, it is plotted as a time series.
        If a dict whose values are Evoked objects, the contents are plotted as
        single time series each and the keys are used as condition labels.
        If a list of Evokeds, the contents are plotted with indices as labels.
        If a [dict/list] of lists, the unweighted mean is plotted as a time
        series and the parametric confidence interval is plotted as a shaded
        area. All instances must have the same shape - channel numbers, time
        points etc.
        If dict, keys must be of type str.
    %(picks_all_data)s

        * If picks is None or a (collection of) data channel types, the
          global field power will be plotted for all data channels.
          Otherwise, picks will be averaged.
        * If multiple channel types are selected, one
          figure will be returned for each channel type.
        * If the selected channels are gradiometers, the signal from
          corresponding (gradiometer) pairs will be combined.

    gfp : bool
        If True, the channel type wise GFP is plotted.
        If None and `picks` is None or a (list of) channel type(s), this is set
        to True.
    colors : list | dict | None
        If a list, will be sequentially used for line colors.
        If a dict, can map evoked keys or '/'-separated (HED) tags to
        conditions.
        For example, if `evokeds` is a dict with the keys "Aud/L", "Aud/R",
        "Vis/L", "Vis/R", `colors` can be `dict(Aud='r', Vis='b')` to map both
        Aud/L and Aud/R to the color red and both Visual conditions to blue.
        If None (default), a sequence of desaturated colors is used.
        If `cmap` is None, `colors` will indicate how each condition is
        colored with reference to its position on the colormap - see `cmap`
        below. In that case, the values of colors must be either integers,
        in which case they will be mapped to colors in rank order; or floats
        between 0 and 1, in which case they will be mapped to percentiles of
        the colormap.
    linestyles : list | dict
        If a list, will be sequentially and repeatedly used for evoked plot
        linestyles.
        If a dict, can map the ``evokeds`` keys or '/'-separated (HED) tags to
        conditions.
        For example, if evokeds is a dict with the keys "Aud/L", "Aud/R",
        "Vis/L", "Vis/R", `linestyles` can be `dict(L='--', R='-')` to map both
        Aud/L and Vis/L to dashed lines and both Right-side conditions to
        straight lines.
    styles : dict | None
        If a dict, keys must map to evoked keys or conditions, and values must
        be a dict of legal inputs to `matplotlib.pyplot.plot`. These
        parameters will be passed to the line plot call of the corresponding
        condition, overriding defaults.
        E.g., if evokeds is a dict with the keys "Aud/L", "Aud/R",
        "Vis/L", "Vis/R", `styles` can be `{"Aud/L": {"linewidth": 1}}` to set
        the linewidth for "Aud/L" to 1. Note that HED ('/'-separated) tags are
        not supported.
    cmap : None | str | tuple
        If not None, plot evoked activity with colors from a color gradient
        (indicated by a str referencing a matplotlib colormap - e.g., "viridis"
        or "Reds").
        If ``evokeds`` is a list and ``colors`` is `None`, the color will
        depend on the list position. If ``colors`` is a list, it must contain
        integers where the list positions correspond to ``evokeds``, and the
        value corresponds to the position on the colorbar.
        If ``evokeds`` is a dict, ``colors`` should be a dict mapping from
        (potentially HED-style) condition tags to numbers corresponding to
        positions on the colorbar - rank order for integers, or floats for
        percentiles. E.g., ::

            evokeds={"cond1/A": ev1, "cond2/A": ev2, "cond3/A": ev3, "B": ev4},
            cmap='viridis', colors=dict(cond1=1 cond2=2, cond3=3),
            linestyles={"A": "-", "B": ":"}

        If ``cmap`` is a tuple of length 2, the first item must be
        a string which will become the colorbar label, and the second one
        must indicate a colormap, e.g. ::

            cmap=('conds', 'viridis'), colors=dict(cond1=1 cond2=2, cond3=3),

    vlines : "auto" | list of float
        A list in seconds at which to plot dashed vertical lines.
        If "auto" and the supplied data includes 0, it is set to [0.]
        and a vertical bar is plotted at time 0. If an empty list is passed,
        no vertical lines are plotted.
    ci : float | callable | None | bool
        If not None and ``evokeds`` is a [list/dict] of lists, a shaded
        confidence interval is drawn around the individual time series. If
        float, a percentile bootstrap method is used to estimate the confidence
        interval and this value determines the CI width. E.g., if this value is
        .95 (the default), the 95%% confidence interval is drawn. If a
        callable, it must take as its single argument an array
        (observations x times) and return the upper and lower confidence bands.
        If None or False, no confidence band is plotted.
        If True, a 95%% bootstrapped confidence interval is drawn.
    truncate_yaxis : bool | str
        If not False, the left y axis spine is truncated to reduce visual
        clutter. If 'max_ticks', the spine is truncated at the minimum and
        maximum ticks. Else, it is truncated to half the max absolute value,
        rounded to .25. Defaults to "max_ticks".
    truncate_xaxis : bool
        If True, the x axis is truncated to span from the first to the last.
        xtick. Defaults to True.
    ylim : dict | None
        ylim for plots (after scaling has been applied). e.g.
        ylim = dict(eeg=[-20, 20])
        Valid keys are eeg, mag, grad, misc. If ``None``, the ylim parameter
        for each channel equals the pyplot default. Defaults to ``None``.
    invert_y : bool
        If True, negative values are plotted up (as is sometimes done
        for ERPs out of tradition). Defaults to False.
    show_sensors: bool | int | str | None
        If not False, channel locations are plotted on a small head circle.
        If int or str, the position of the axes (forwarded to
        ``mpl_toolkits.axes_grid1.inset_locator.inset_axes``).
        If None, defaults to True if ``gfp`` is False, else to False.
    show_legend : bool | str | int
        If not False, show a legend. If int or str, it is the position of the
        legend axes (forwarded to
        ``mpl_toolkits.axes_grid1.inset_locator.inset_axes``).
    split_legend : bool
        If True, the legend shows color and linestyle separately; `colors` must
        not be None. Defaults to True if ``cmap`` is not None, else defaults to
        False.
    axes : None | `matplotlib.axes.Axes` instance | list of `axes` | "topo"
        What axes to plot to. If None, a new axes is created.
        If "topo", separately for each channel type, a new figure is created
        with one axis for each channel in a topographical layout. In this
        case, `gfp` is ignored.
        When plotting multiple channel types, can also be a list of axes, one
        per channel type.
    title : None | str
        If str, will be plotted as figure title. If None, the channel names
        will be shown.
    show : bool
        If True, show the figure.

    Returns
    -------
    fig : Figure | list of Figure
        The figure(s) in which the plot is drawn. When plotting multiple
        channel types, a list of figures, one for each channel type is
        returned.

    Notes
    -----
    When multiple channels are passed, this function combines them all, to
    get one time course for each condition. If gfp is True it combines
    channels using global field power (GFP) computation, else it is taking
    a plain mean.

    This function is useful for comparing multiple ER[P/F]s - e.g., for
    multiple conditions - at a specific location.

    It can plot:

    - a simple :class:`mne.Evoked` object,
    - a list or dict of :class:`mne.Evoked` objects (e.g., for multiple
      conditions),
    - a list or dict of lists of :class:`mne.Evoked` (e.g., for multiple
      subjects in multiple conditions).

    In the last case, it can show a confidence interval (across e.g. subjects)
    using parametric or bootstrap estimation.

    When ``picks`` includes more than one planar gradiometer, the planar
    gradiometers are combined with RMSE. For example data from a
    VectorView system with 204 gradiometers will be transformed to
    102 channels.
    """
    import matplotlib.pyplot as plt

    evokeds, colors = _format_evokeds_colors(evokeds, cmap, colors)
    conditions = sorted(list(evokeds.keys()))

    # get and set a few limits and variables (times, channels, units)
    one_evoked = evokeds[conditions[0]][0]
    times = one_evoked.times
    info = one_evoked.info
    tmin, tmax = times[0], times[-1]

    if vlines == "auto":
        vlines = [0.] if (tmin < 0 < tmax) else []
    _validate_type(vlines, (list, tuple), "vlines", "list or tuple")

    # default to plotting the GFP if all picks are channel type strs
    if gfp is None and (picks is None or (
            (picks == 'meg' or picks in _DATA_CH_TYPES_SPLIT) or
            (isinstance(picks, Iterable) and
             all(pick == 'meg' or pick in _DATA_CH_TYPES_SPLIT
                 for pick in picks)))):
        gfp = True

    picks_was_str_title_was_none = False
    picks = [] if picks is None else picks
    if title is None and picks in _DATA_CH_TYPES_SPLIT:
        title = _handle_default('titles')[picks]
        if gfp:
            picks_was_str_title_was_none = True

    picks = _picks_to_idx(info, picks, allow_empty=True)
    if len(picks) == 0:
        if axes != "topo" or gfp is not False:
            logger.info("No picks, plotting the GFP ...")
            gfp = True
        picks = _pick_data_channels(info, with_ref_meg=False)

    if picks_was_str_title_was_none and gfp:
        title += " (GFP)"

    _validate_type(picks, (list, np.ndarray), "picks",
                   "list or np.array of integers")
    for entry in picks:
        _validate_type(entry, 'int', "entries of picks", "integers")

    if len(picks) == 0:
        raise ValueError("No valid channels were found to plot the GFP. "
                         "Use 'picks' instead to select them manually.")

    if ylim is None:
        ylim = dict()

    do_topo = False
    if axes == "topo":
        gfp = False
        do_topo = True
        show_sensors = False
        if show_legend is None:
            show_legend = "lower right"
        if len(picks) > 70:
            logger.info("Warning: plotting to a topographical layout with "
                        "> 70 sensors. This can be extremely slow. For a "
                        "Faster alternative, consider using "
                        ":func:`mne.viz.plot_topo`, which is optimized for "
                        "speed.")

    # deal with picks: infer indices and names
    if gfp:
        if show_sensors is None:
            show_sensors = False  # don't show sensors for GFP
        ch_names = ['Global Field Power']
        if len(picks) < 2:
            raise ValueError("Cannot compute GFP for fewer than 2 channels, "
                             "please pick more than %d channels." % len(picks))
    else:
        if show_sensors is None:
            show_sensors = True  # show sensors when not doing GFP
        ch_names = [one_evoked.ch_names[pick] for pick in picks]

    picks_by_types = channel_indices_by_type(info, picks)
    # keep only channel types for which there is a channel:
    ch_types = [t for t in picks_by_types if len(picks_by_types[t]) > 0]

    # let's take care of axis and figs
    if not do_topo:
        if axes is not None:
            if not isinstance(axes, list):
                axes = [axes]
                _validate_if_list_of_axes(axes, obligatory_len=len(ch_types))
        else:
            axes = (plt.subplots(figsize=(8, 6))[1]
                    for _ in range(len(ch_types)))
    else:
        axes = ["topo"] * len(ch_types)

    if len(ch_types) > 1:
        logger.info("Multiple channel types selected, returning one figure "
                    "per type.")
        figs = list()
        for t, ax in zip(ch_types, axes):
            picks_ = picks_by_types[t]
            title_ = "GFP, " + t if (title is None and gfp) else title
            figs.append(plot_compare_evokeds(
                evokeds, picks=picks_, gfp=gfp, colors=colors,
                linestyles=linestyles, styles=styles, vlines=vlines, ci=ci,
                truncate_yaxis=truncate_yaxis, ylim=ylim, invert_y=invert_y,
                show_legend=show_legend, show_sensors=show_sensors,
                axes=ax, title=title_, split_legend=split_legend, show=show))
        return figs

    # From now on there is only 1 channel type
    assert len(ch_types) == 1
    ch_type = ch_types[0]

    all_positive = gfp
    pos_picks = picks  # keep locations to pick for plotting
    if ch_type == "grad" and len(picks) > 1:
        logger.info('Combining all planar gradiometers with RMSE.')
        pos_picks, _ = _grad_pair_pick_and_name(one_evoked.info, picks)
        pos_picks = pos_picks[::2]
        all_positive = True
        for cond, this_evokeds in evokeds.items():
            evokeds[cond] = [_combine_grad(e, picks) for e in this_evokeds]
        ch_names = evokeds[cond][0].ch_names
        picks = range(len(ch_names))

    info = pick_info(info, pos_picks, True)
    all_ch_names = ch_names if ch_type == 'grad' else info['ch_names']
    if do_topo:
        from .topo import iter_topography
        fig = plt.figure(figsize=(18, 14))

        def click_func(
                ax_, pick_, evokeds=evokeds, gfp=gfp, colors=colors,
                linestyles=linestyles, styles=styles, cmap=cmap, vlines=vlines,
                ci=ci, truncate_yaxis=truncate_yaxis,
                truncate_xaxis=truncate_xaxis, ylim=ylim, invert_y=invert_y,
                show_sensors=show_sensors, show_legend=show_legend,
                split_legend=split_legend, picks=picks):
            plot_compare_evokeds(
                evokeds=evokeds, gfp=gfp, colors=colors, linestyles=linestyles,
                styles=styles, cmap=cmap, vlines=vlines, ci=ci,
                truncate_yaxis=truncate_yaxis, truncate_xaxis=truncate_xaxis,
                ylim=ylim, invert_y=invert_y, show_sensors=show_sensors,
                show_legend=show_legend, split_legend=split_legend,
                show=True, picks=picks[pick_], axes=ax_)

        layout = find_layout(info)
        # shift everything to the right by 15% of one axes width
        layout.pos[:, 0] += layout.pos[0, 2] * .15
        layout.pos[:, 1] += layout.pos[0, 3] * .15
        # fixme: prevent having to loop over the axes multiple times
        axes = list(iter_topography(
            info, layout=layout, on_pick=click_func,
            fig=fig, fig_facecolor='w', axis_facecolor='w',
            axis_spinecolor='k', layout_scale=.925, legend=True))
    del info

    ymin, ymax = ylim.get(ch_type, [None, None])

    scaling = _handle_default("scalings")[ch_type]
    unit = _handle_default("units")[ch_type]

    if (ymin is None) and all_positive:
        ymin = 0.  # 'grad' and GFP are plotted as all-positive

    # title
    title = _set_title_multiple_electrodes(
        title, ('gfp' if gfp else 'average'), ch_names, ch_type=ch_type)

    colors_for_split_legend, styles_for_split_legend = (
        deepcopy(colors), deepcopy(linestyles))
    (styles, colors, the_colors, color_conds, color_order, colors_are_float,
     cmap_label, cmap) = _finish_styles_plot_comp_evoked(
         styles, colors, linestyles, conditions, cmap)
    # We now have a 'styles' dict with one entry per condition, specifying at
    # least color and linestyles.

    ci_fun = _get_ci_function_for_evokeds(ci)

    # if we have a dict/list of lists, we compute the grand average and the CI
    # (per sensor if topo, otherwise aggregating over sensors)
    if not do_topo:
        axes = [(ax, 0) for ax in axes]
        if not (isinstance(picks, Iterable) and
                isinstance(picks[0], Iterable)):
            picks = [picks]  # because we iterate over it
    else:
        picks = list(picks)

    all_data, all_cis = [], []
    any_positive, any_negative = False, False
    for picks_, (ax, idx) in zip(picks, axes):
        data_dict, ci_dict = _calculate_ci_and_mean(
            evokeds, conditions, scaling, picks_, ci_fun, gfp)
        # we now have dicts for data ('evokeds' - grand averaged Evoked's)
        # and the CI ('ci_array') with cond name labels
        all_data.append(data_dict)
        all_cis.append(ci_dict)
    del evokeds

    if do_topo:
        picks.append(-1)  # so the loops don't terminate prematurely

    # add empty data (all zeros) for the legend axis
    all_data.append({cond: np.zeros(d.shape) for cond, d in data_dict.items()})
    all_cis.append({cond: np.zeros(np.array(d).shape)
                    for cond, d in ci_dict.items()})

    for picks_, (ax, idx), data, cis in zip(picks, axes, all_data, all_cis):
        if do_topo:
            title = all_ch_names[idx]
        do_ci = ci_fun is not None
        if idx == -1:
            do_ci = False
        any_positive_, any_negative_ = _plot_compare_evokeds(
            ax, data, conditions, times, do_ci, cis, styles,
            title, all_positive, do_topo)
        if any_positive_:
            any_positive = True
        if any_negative_:
            any_negative = True

    if do_topo:
        if ci_fun is not None:
            all_data = all_cis
        data = np.array([list(d.values()) for d in all_data])
        ymin, ymax = _setup_vmin_vmax(data, ymin, ymax)

    # ylims
    for ax_, idx in axes:
        skip_axlabel = do_topo and (idx != -1)
        _set_ylims_plot_compare_evokeds(
            ax_, any_positive, any_negative, ymin, ymax,
            truncate_yaxis, truncate_xaxis, invert_y, vlines, tmin, tmax, unit,
            skip_axlabel=skip_axlabel)
        ax_.patch.set_alpha(0)
        if do_topo and idx != -1:
            ax_.set_yticklabels([])
            ax_.set_xticklabels([])

    # 2 legends.
    # a head plot showing the sensors that are being plotted
    if show_sensors:
        _validate_type(show_sensors, (np.int, bool, str, type(None)),
                       "show_sensors", "numeric, str, None or bool")
        if not _check_ch_locs(np.array(one_evoked.info['chs'])[pos_picks]):
            warn("Cannot find channel coordinates in the supplied Evokeds. "
                 "Not showing channel locations.")
        else:
            _evoked_sensor_legend(one_evoked.info, pos_picks, ymin, ymax,
                                  show_sensors, ax_)

    # condition legend
    _evoked_condition_legend(
        conditions, show_legend, split_legend, cmap, colors_are_float,
        the_colors, colors if not split_legend else colors_for_split_legend,
        color_conds, color_order, cmap_label,
        linestyles if not split_legend else styles_for_split_legend, ax_,
        do_topo)

    plt_show(show)
    return ax.figure
