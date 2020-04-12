# -*- coding: utf-8 -*-
"""Functions to plot evoked M/EEG data (besides topographies)."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

from copy import deepcopy
from functools import partial
from numbers import Integral

import numpy as np

from ..io.pick import (channel_type,
                       _VALID_CHANNEL_TYPES, channel_indices_by_type,
                       _DATA_CH_TYPES_SPLIT, _pick_inst, _get_channel_types,
                       _PICK_TYPES_DATA_DICT, _picks_to_idx, pick_info)
from ..defaults import _handle_default
from .utils import (_draw_proj_checkbox, tight_layout, _check_delayed_ssp,
                    plt_show, _process_times, DraggableColorbar, _setup_cmap,
                    _setup_vmin_vmax, _check_cov, _make_combine_callable,
                    _validate_if_list_of_axes, _triage_rank_sss,
                    _connection_line, _get_color_list, _setup_ax_spines,
                    _setup_plot_projector, _prepare_joint_axes,
                    _set_title_multiple_electrodes, _check_time_unit,
                    _plot_masked_image, _trim_ticks)
from ..utils import (logger, _clean_names, warn, _pl, verbose, _validate_type,
                     _check_if_nan, _check_ch_locs, fill_doc, _is_numeric)

from .topo import _plot_evoked_topo
from .topomap import (_prepare_topomap_plot, plot_topomap, _get_pos_outlines,
                      _draw_outlines, _prepare_topomap, _set_contour_locator,
                      _check_sphere, _make_head_outlines)
from ..channels.layout import _pair_grad_sensors, find_layout


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
                        psd=False, time_unit='s', sphere=None):
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
        picks, pos, merge_channels, _, ch_type, this_sphere, clip_origin = \
            _prepare_topomap_plot(info, ch_type, sphere=sphere)
        outlines = _make_head_outlines(this_sphere, pos, 'head', clip_origin)
        if len(pos) < 2:
            fig.delaxes(axarr[0][idx])
            continue
        this_data = data[picks, minidx:maxidx]
        if merge_channels:
            from ..channels.layout import _merge_ch_data
            method = 'mean' if psd else 'rms'
            this_data, _ = _merge_ch_data(this_data, ch_type, [],
                                          method=method)
            title = '%s %s' % (ch_type, method.upper())
        else:
            title = ch_type
        this_data = np.average(this_data, axis=1)
        axarr[0][idx].set_title(title)
        vmin = min(this_data) if psd else None
        vmax = max(this_data) if psd else None  # All negative for dB psd.
        cmap = 'Reds' if psd else None
        plot_topomap(this_data, pos, cmap=cmap, vmin=vmin, vmax=vmax,
                     axes=axarr[0][idx], show=False, sphere=this_sphere,
                     outlines=outlines)

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
                 show_names=False, group_by=None, sphere=None):
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
            titles = ({channel_type(evoked.info, idx): sel
                       for idx in group_by[sel]} if titles is None else titles)
            _plot_evoked(evoked, group_by[sel], exclude, unit, show, ylim,
                         proj, xlim, hline, units, scalings, titles,
                         ax, plot_type, cmap=cmap, gfp=gfp,
                         window_title=window_title,
                         set_tight_layout=set_tight_layout,
                         selectable=selectable, noise_cov=noise_cov,
                         colorbar=colorbar, mask=mask,
                         mask_style=mask_style, mask_cmap=mask_cmap,
                         mask_alpha=mask_alpha, time_unit=time_unit,
                         show_names=show_names,
                         sphere=sphere)
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

    types = np.array(_get_channel_types(info, picks), np.unicode)
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
                    time_unit=time_unit, sphere=sphere)
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
                line_alpha, nave, time_unit, sphere):
    """Plot data as butterfly plot."""
    from matplotlib import patheffects, pyplot as plt
    from matplotlib.widgets import SpanSelector
    assert len(axes) == len(ch_types_used)
    texts = list()
    idxs = list()
    lines = list()
    sphere = _check_sphere(sphere, info)
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
                                           ax, sphere)
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
                        ax.plot(times, D[ch_idx], picker=True,
                                zorder=z + 1 if spatial_colors is True else 1,
                                color=colors[ch_idx], alpha=line_alpha,
                                linewidth=0.5)[0])
                    line_list[-1].set_pickradius(3.)

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
                                 fontweight='bold', alpha=0,
                                 clip_on=True))

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
                                        psd=psd, time_unit=time_unit,
                                        sphere=sphere)
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


def _handle_spatial_colors(colors, info, idx, ch_type, psd, ax, sphere):
    """Set up spatial colors."""
    used_nm = np.array(_clean_names(info['ch_names']))[idx]
    # find indices for bads
    bads = [np.where(used_nm == bad)[0][0] for bad in info['bads'] if bad in
            used_nm]
    pos, outlines = _get_pos_outlines(info, idx, sphere=sphere)
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

    # ignore xlim='tight'; happens automatically with `extent` in imshow
    xlim = None if xlim == 'tight' else xlim
    if xlim is not None:
        ax.set_xlim(xlim)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_title(ch_unit)
        if cmap[1]:
            ax.CB = DraggableColorbar(cbar, im)

    ylabel = 'Channels' if show_names else 'Channel (index)'
    t = titles[this_type] + ' (%d channel%s' % (len(data), _pl(data)) + t_end
    ax.set(ylabel=ylabel, xlabel='Time (%s)' % (time_unit,), title=t)
    _add_nave(ax, nave)

    yticks = np.arange(len(picks))
    if show_names != 'all':
        yticks = np.intersect1d(np.round(ax.get_yticks()).astype(int), yticks)
    yticklabels = np.array(ch_names)[picks] if show_names else np.array(picks)
    ax.set(yticks=yticks, yticklabels=yticklabels[yticks])


@verbose
def plot_evoked(evoked, picks=None, exclude='bads', unit=True, show=True,
                ylim=None, xlim='tight', proj=False, hline=None, units=None,
                scalings=None, titles=None, axes=None, gfp=False,
                window_title=None, spatial_colors=False, zorder='unsorted',
                selectable=True, noise_cov=None, time_unit='s', sphere=None,
                verbose=None):
    """Plot evoked data using butterfly plots.

    Left click to a line shows the channel name. Selecting an area by clicking
    and holding left mouse button plots a topographic map of the painted area.

    .. note:: If bad channels are not excluded they are shown in red.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    %(picks_all)s
    exclude : list of str | 'bads'
        Channels names to exclude from being shown. If 'bads', the
        bad channels are excluded.
    unit : bool
        Scale plot with channel (SI) unit.
    show : bool
        Show figure if True.
    ylim : dict | None
        Y limits for plots (after scaling has been applied). e.g.
        ylim = dict(eeg=[-20, 20])
        Valid keys are eeg, mag, grad, misc. If None, the ylim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        X limits for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    hline : list of float | None
        The values at which to show an horizontal line.
    units : dict | None
        The units of the channel types used for axes labels. If None,
        defaults to ``dict(eeg='µV', grad='fT/cm', mag='fT')``.
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
    %(topomap_sphere_auto)s
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
        time_unit=time_unit, sphere=sphere)


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
    layout_scale : float
        Scaling factor for adjusting the relative size of the layout
        on the canvas.
    color : list of color | color | None
        Everything matplotlib accepts to specify colors. If not list-like,
        the color specified will be repeated. If None, colors are
        automatically drawn.
    border : str
        Matplotlib borders style to be used for each sensor plot.
    ylim : dict | None
        Y limits for plots (after scaling has been applied). The value
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
    legend : bool | int | str | tuple
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
        Images of evoked responses at sensor locations.
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
                             font_color=font_color,
                             merge_channels=merge_grads,
                             legend=legend, axes=axes, show=show,
                             noise_cov=noise_cov)


@fill_doc
def plot_evoked_image(evoked, picks=None, exclude='bads', unit=True,
                      show=True, clim=None, xlim='tight', proj=False,
                      units=None, scalings=None, titles=None, axes=None,
                      cmap='RdBu_r', colorbar=True, mask=None,
                      mask_style=None, mask_cmap="Greys", mask_alpha=.25,
                      time_unit='s', show_names="auto", group_by=None,
                      sphere=None):
    """Plot evoked data as images.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
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
        Color limits for plots (after scaling has been applied). e.g.
        ``clim = dict(eeg=[-20, 20])``.
        Valid keys are eeg, mag, grad, misc. If None, the clim parameter
        for each channel equals the pyplot default.
    xlim : 'tight' | tuple | None
        X limits for plots.
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be shown.
    units : dict | None
        The units of the channel types used for axes labels. If None,
        defaults to ``dict(eeg='µV', grad='fT/cm', mag='fT')``.
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
    mask_style : None | 'both' | 'contour' | 'mask'
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
    show_names : bool | 'auto' | 'all'
        Determines if channel names should be plotted on the y axis. If False,
        no names are shown. If True, ticks are set automatically by matplotlib
        and the corresponding channel names are shown. If "all", all channel
        names are shown. If "auto", is set to False if `picks` is ``None``,
        to ``True`` if ``picks`` contains 25 or more entries, or to "all"
        if ``picks`` contains fewer than 25 entries.
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
    %(topomap_sphere_auto)s

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
                        group_by=group_by, sphere=sphere)


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
                      sphere=None, verbose=None):
    """Plot whitened evoked response.

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
    %(topomap_sphere_auto)s
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
                              time_unit=time_unit, sphere=sphere)


def _plot_evoked_white(evoked, noise_cov, scalings, rank, show, time_unit,
                       sphere):
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
    ch_types = _get_channel_types(info, unique=True, only_data_chs=True)

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
            if len(_get_channel_types(ev_.info, unique=True)) > 1:
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
                       spatial_colors=True, zorder='std',
                       sphere=None)
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

def _check_loc_legal(loc, what='your choice', default=1):
    """Check if loc is a legal location for MPL subordinate axes."""
    true_default = {"legend": 2, "show_sensors": 1}.get(what, default)
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


def _validate_style_keys_pce(styles, conditions, tags):
    """Validate styles dict keys for plot_compare_evokeds."""
    styles = deepcopy(styles)
    if not set(styles).issubset(tags.union(conditions)):
        raise ValueError('The keys in "styles" ({}) must match the keys in '
                         '"evokeds" ({}).'.format(list(styles), conditions))
    # make sure all the keys are in there
    for cond in conditions:
        if cond not in styles:
            styles[cond] = dict()
        # deal with matplotlib's synonymous handling of "c" and "color" /
        # "ls" and "linestyle" / "lw" and "linewidth"
        elif 'c' in styles[cond]:
            styles[cond]['color'] = styles[cond].pop('c')
        elif 'ls' in styles[cond]:
            styles[cond]['linestyle'] = styles[cond].pop('ls')
        elif 'lw' in styles[cond]:
            styles[cond]['linewidth'] = styles[cond].pop('lw')
        # transfer styles from partial-matched entries
        for tag in cond.split('/'):
            if tag in styles:
                styles[cond].update(styles[tag])
    # remove the (now transferred) partial-matching style entries
    for key in list(styles):
        if key not in conditions:
            del styles[key]
    return styles


def _validate_colors_pce(colors, cmap, conditions, tags):
    """Check and assign colors for plot_compare_evokeds."""
    err_suffix = ''
    if colors is None:
        if cmap is None:
            colors = _get_color_list()
            err_suffix = ' in the default color cycle'
        else:
            colors = list(range(len(conditions)))
    # convert color list to dict
    if isinstance(colors, (list, tuple, np.ndarray)):
        if len(conditions) > len(colors):
            raise ValueError('Trying to plot {} conditions, but there are only'
                             ' {} colors{}. Please specify colors manually.'
                             .format(len(conditions), len(colors), err_suffix))
        colors = dict(zip(conditions, colors))
    # should be a dict by now...
    if not isinstance(colors, dict):
        raise TypeError('"colors" must be a dict, list, or None; got {}.'
                        .format(type(colors).__name__))
    # validate color dict keys
    if not set(colors).issubset(tags.union(conditions)):
        raise ValueError('If "colors" is a dict its keys ({}) must '
                         'match the keys/conditions in "evokeds" ({}).'
                         .format(list(colors), conditions))
    # validate color dict values
    color_vals = list(colors.values())
    all_numeric = all(_is_numeric(_color) for _color in color_vals)
    if cmap is not None and not all_numeric:
        raise TypeError('if "cmap" is specified, then "colors" must be '
                        'None or a (list or dict) of (ints or floats); got {}.'
                        .format(', '.join(color_vals)))
    # convert provided ints to sequential, rank-ordered ints
    all_int = all([isinstance(_color, Integral) for _color in color_vals])
    if all_int:
        colors = deepcopy(colors)
        ranks = {val: ix for ix, val in enumerate(sorted(set(color_vals)))}
        for key, orig_int in colors.items():
            colors[key] = ranks[orig_int]
        # if no cmap, convert color ints to real colors
        if cmap is None:
            color_list = _get_color_list()
            for cond, color_int in colors.items():
                colors[cond] = color_list[color_int]
    # recompute color_vals as a sorted set (we'll need it that way later)
    color_vals = set(colors.values())
    if all_numeric:
        color_vals = sorted(color_vals)
    return colors, color_vals


def _validate_cmap_pce(cmap, colors, color_vals):
    """Check and assign colormap for plot_compare_evokeds."""
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Colormap
    all_int = all([isinstance(_color, Integral) for _color in color_vals])
    lut = len(color_vals) if all_int else None
    colorbar_title = ''
    if isinstance(cmap, (list, tuple, np.ndarray)) and len(cmap) == 2:
        colorbar_title, cmap = cmap
    if isinstance(cmap, str):
        cmap = get_cmap(cmap, lut=lut)
    elif isinstance(cmap, Colormap) and all_int:
        cmap = cmap._resample(lut)
    return cmap, colorbar_title


def _validate_linestyles_pce(linestyles, conditions, tags):
    """Check and assign linestyles for plot_compare_evokeds."""
    # make linestyles a list if it's not defined
    if linestyles is None:
        linestyles = [None] * len(conditions)  # will get changed to defaults
    # convert linestyle list to dict
    if isinstance(linestyles, (list, tuple, np.ndarray)):
        if len(conditions) > len(linestyles):
            raise ValueError('Trying to plot {} conditions, but there are '
                             'only {} linestyles. Please specify linestyles '
                             'manually.'
                             .format(len(conditions), len(linestyles)))
        linestyles = dict(zip(conditions, linestyles))
    # should be a dict by now...
    if not isinstance(linestyles, dict):
        raise TypeError('"linestyles" must be a dict, list, or None; got {}.'
                        .format(type(linestyles).__name__))
    # validate linestyle dict keys
    if not set(linestyles).issubset(tags.union(conditions)):
        raise ValueError('If "linestyles" is a dict its keys ({}) must '
                         'match the keys/conditions in "evokeds" ({}).'
                         .format(list(linestyles), conditions))
    # normalize linestyle values (so we can accurately count unique linestyles
    # later). See https://github.com/matplotlib/matplotlib/blob/master/matplotlibrc.template#L131-L133  # noqa
    linestyle_map = {'solid': (0, ()),
                     'dotted': (0, (1., 1.65)),
                     'dashed': (0, (3.7, 1.6)),
                     'dashdot': (0, (6.4, 1.6, 1., 1.6)),
                     '-': (0, ()),
                     ':': (0, (1., 1.65)),
                     '--': (0, (3.7, 1.6)),
                     '-.': (0, (6.4, 1.6, 1., 1.6))}
    for cond, _ls in linestyles.items():
        linestyles[cond] = linestyle_map.get(_ls, _ls)
    return linestyles


def _populate_style_dict_pce(condition, condition_styles, style_name,
                             style_dict, cmap):
    """Transfer styles into condition_styles dict for plot_compare_evokeds."""
    defaults = dict(color='gray', linestyle=(0, ()))  # (0, ()) == 'solid'
    # if condition X doesn't yet have style Y defined:
    if condition_styles.get(style_name, None) is None:
        # check the style dict for the full condition name
        try:
            condition_styles[style_name] = style_dict[condition]
        # if it's not in there, try the slash-separated condition tags
        except KeyError:
            for tag in condition.split('/'):
                try:
                    condition_styles[style_name] = style_dict[tag]
                # if the tag's not in there, assign a default value (but also
                # continue looping in search of a tag that *is* in there)
                except KeyError:
                    condition_styles[style_name] = defaults[style_name]
                # if we found a valid tag, keep track of it for colorbar
                # legend purposes, and also stop looping (so we don't overwrite
                # a valid tag's style with an invalid tag → default style)
                else:
                    if style_name == 'color' and cmap is not None:
                        condition_styles['cmap_label'] = tag
                    break
    return condition_styles


def _handle_styles_pce(styles, linestyles, colors, cmap, conditions):
    """Check and assign styles for plot_compare_evokeds."""
    styles = deepcopy(styles)
    # validate style dict structure (doesn't check/assign values yet)
    tags = set(tag for cond in conditions for tag in cond.split('/'))
    if styles is None:
        styles = {cond: dict() for cond in conditions}
    styles = _validate_style_keys_pce(styles, conditions, tags)
    # validate color dict
    colors, color_vals = _validate_colors_pce(colors, cmap, conditions, tags)
    all_int = all([isinstance(_color, Integral) for _color in color_vals])
    # instantiate cmap
    cmap, colorbar_title = _validate_cmap_pce(cmap, colors, color_vals)
    # validate linestyles
    linestyles = _validate_linestyles_pce(linestyles, conditions, tags)

    # prep for colorbar tick handling
    colorbar_ticks = None if cmap is None else dict()
    # array mapping color integers (indices) to tick locations (array values)
    tick_locs = np.linspace(0, 1, 2 * len(color_vals) + 1)[1::2]

    # transfer colors/linestyles dicts into styles dict; fall back on defaults
    color_and_linestyle = dict(color=colors, linestyle=linestyles)
    for cond, cond_styles in styles.items():
        for _name, _style in color_and_linestyle.items():
            cond_styles = _populate_style_dict_pce(cond, cond_styles, _name,
                                                   _style, cmap)
        # convert numeric colors into cmap color values; store colorbar ticks
        if cmap is not None:
            color_number = cond_styles['color']
            cond_styles['color'] = cmap(color_number)
            tick_loc = tick_locs[color_number] if all_int else color_number
            key = cond_styles.pop('cmap_label', cond)
            colorbar_ticks[key] = tick_loc

    return styles, linestyles, colors, cmap, colorbar_title, colorbar_ticks


def _evoked_sensor_legend(info, picks, ymin, ymax, show_sensors, ax,
                          sphere):
    """Show sensor legend (location of a set of sensors on the head)."""
    if show_sensors is True:
        ymin, ymax = np.abs(ax.get_ylim())
        show_sensors = "lower right" if ymin > ymax else "upper right"

    pos, outlines = _get_pos_outlines(info, picks, sphere=sphere)
    show_sensors = _check_loc_legal(show_sensors, "show_sensors")
    _plot_legend(pos, ["k"] * len(picks), ax, list(), outlines,
                 show_sensors, size=25)


def _draw_colorbar_pce(ax, colors, cmap, colorbar_title, colorbar_ticks):
    """Draw colorbar for plot_compare_evokeds."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.transforms import Bbox
    # create colorbar axes
    orig_bbox = ax.get_position()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cax.yaxis.tick_right()
    cb = ColorbarBase(cax, cmap=cmap, norm=None, orientation='vertical')
    cb.set_label(colorbar_title)
    # handle ticks
    ticks = sorted(set(colorbar_ticks.values()))
    ticklabels = [''] * len(ticks)
    for label, tick in colorbar_ticks.items():
        idx = ticks.index(tick)
        if len(ticklabels[idx]):  # handle labels with the same color/location
            ticklabels[idx] = '\n'.join([ticklabels[idx], label])
        else:
            ticklabels[idx] = label
    assert all(len(label) for label in ticklabels)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    # shrink colorbar if discrete colors
    color_vals = set(colors.values())
    if all([isinstance(_color, Integral) for _color in color_vals]):
        fig = ax.get_figure()
        fig.canvas.draw()
        fig_aspect = np.divide(*fig.get_size_inches())
        new_bbox = ax.get_position()
        cax_width = 0.75 * (orig_bbox.xmax - new_bbox.xmax)
        # add extra space for multiline colorbar labels
        h_mult = max(2, max([len(label.split('\n')) for label in ticklabels]))
        cax_height = len(color_vals) * h_mult * cax_width / fig_aspect
        x0 = orig_bbox.xmax - cax_width
        y0 = (new_bbox.ymax + new_bbox.ymin - cax_height) / 2
        x1 = orig_bbox.xmax
        y1 = y0 + cax_height
        new_bbox = Bbox([[x0, y0], [x1, y1]])
        cax.set_axes_locator(None)
        cax.set_position(new_bbox)


def _draw_legend_pce(legend, split_legend, styles, linestyles, colors, cmap,
                     do_topo, ax):
    """Draw legend for plot_compare_evokeds."""
    import matplotlib.lines as mlines
    lines = list()
    # triage
    if split_legend is None:
        split_legend = cmap is not None
    n_colors = len(set(colors.values()))
    n_linestyles = len(set(linestyles.values()))
    draw_styles = cmap is None and not split_legend
    draw_colors = cmap is None and split_legend and n_colors > 1
    draw_linestyles = (cmap is None or split_legend) and n_linestyles > 1
    # create the fake lines for the legend
    if draw_styles:
        for label, cond_styles in styles.items():
            line = mlines.Line2D([], [], label=label, **cond_styles)
            lines.append(line)
    else:
        if draw_colors:
            for label, color in colors.items():
                line = mlines.Line2D([], [], label=label, linestyle='solid',
                                     color=color)
                lines.append(line)
        if draw_linestyles:
            for label, linestyle in linestyles.items():
                line = mlines.Line2D([], [], label=label, linestyle=linestyle,
                                     color='black')
                lines.append(line)
    # legend params
    ncol = 1 + (len(lines) // 5)
    loc = _check_loc_legal(legend, 'legend')
    legend_params = dict(loc=loc, frameon=True, ncol=ncol)
    # special placement (above dedicated legend axes) in topoplot
    if do_topo and isinstance(legend, bool):
        legend_params.update(loc='lower right', bbox_to_anchor=(1, 1))
    # draw the legend
    if any([draw_styles, draw_colors, draw_linestyles]):
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, **legend_params)


def _draw_axes_pce(ax, ymin, ymax, truncate_yaxis, truncate_xaxis, invert_y,
                   vlines, tmin, tmax, unit, skip_axlabel=True):
    """Position, draw, and truncate axes for plot_compare_evokeds."""
    # avoid matplotlib errors
    if ymin == ymax:
        ymax += 1e-15
    if tmin == tmax:
        tmax += 1e-9
    ax.set_xlim(tmin, tmax)
    # for dark backgrounds:
    ax.patch.set_alpha(0)
    if not np.isfinite([ymin, ymax]).all():  # nothing plotted
        return
    ax.set_ylim(ymin, ymax)
    ybounds = (ymin, ymax)
    # determine ymin/ymax for spine truncation
    trunc_y = True if truncate_yaxis == 'auto' else truncate_yaxis
    if truncate_yaxis:
        if isinstance(truncate_yaxis, bool):
            # truncate to half the max abs. value and round to a nice-ish
            # number. ylims are already symmetric about 0 or have a lower bound
            # of 0, so div. by 2 should suffice.
            ybounds = np.array([ymin, ymax]) / 2.
            precision = 0.25
            ybounds = np.round(ybounds / precision) * precision
        elif truncate_yaxis == 'auto':
            # truncate to existing max/min ticks
            ybounds = _trim_ticks(ax.get_yticks(), ymin, ymax)[[0, -1]]
        else:
            raise ValueError('"truncate_yaxis" must be bool or '
                             '"auto", got {}'.format(truncate_yaxis))
    _setup_ax_spines(ax, vlines, tmin, tmax, ybounds[0], ybounds[1], invert_y,
                     unit, truncate_xaxis, trunc_y, skip_axlabel)


def _get_data_and_ci(evoked, combine, combine_func, picks, scaling=1,
                     ci_fun=None):
    """Compute (sensor-aggregated, scaled) time series and possibly CI."""
    picks = np.array(picks).flatten()
    # apply scalings
    data = np.array([evk.data[picks] * scaling for evk in evoked])
    # combine across sensors
    if combine is not None:
        logger.info('combining channels using "{}"'.format(combine))
        data = combine_func(data)
    # get confidence band
    if ci_fun is not None:
        ci = ci_fun(data)
    # get grand mean across evokeds
    data = np.mean(data, axis=0)
    _check_if_nan(data)
    return (data,) if ci_fun is None else (data, ci)


def _get_ci_function_pce(ci, do_topo=False):
    """Get confidence interval function for plot_compare_evokeds."""
    if ci is None:
        return None
    elif callable(ci):
        return ci
    elif isinstance(ci, bool) and not ci:
        return None
    elif isinstance(ci, bool):
        ci = 0.95
    if isinstance(ci, float):
        from ..stats import _ci
        method = 'parametric' if do_topo else 'bootstrap'
        return partial(_ci, ci=ci, method=method)
    else:
        raise TypeError('"ci" must be None, bool, float or callable, got {}'
                        .format(type(ci).__name__))


def _plot_compare_evokeds(ax, data_dict, conditions, times, ci_dict, styles,
                          title, all_positive, topo):
    """Plot evokeds (to compare them; with CIs) based on a data_dict."""
    for condition in conditions:
        # plot the actual data ('dat') as a line
        dat = data_dict[condition].T
        ax.plot(times, dat, zorder=1000, label=condition, clip_on=False,
                **styles[condition])
        # plot the confidence interval if available
        if ci_dict.get(condition, None) is not None:
            ci_ = ci_dict[condition]
            ax.fill_between(times, ci_[0].flatten(), ci_[1].flatten(),
                            zorder=9, color=styles[condition]['color'],
                            alpha=0.3, clip_on=False)
    if topo:
        ax.text(-.1, 1, title, transform=ax.transAxes)
    else:
        ax.set_title(title)


def _title_helper_pce(title, picked_types, picks, ch_names, combine):
    """Format title for plot_compare_evokeds."""
    if title is None:
        title = (_handle_default('titles').get(picks, None) if picked_types
                 else _set_title_multiple_electrodes(title, combine, ch_names))
    # add the `combine` modifier
    do_combine = picked_types or len(ch_names) > 1
    if (title is not None and len(title) and isinstance(combine, str) and
            do_combine):
        _comb = combine.upper() if combine == 'gfp' else combine
        _comb = 'std. dev.' if _comb == 'std' else _comb
        title += ' ({})'.format(_comb)
    return title


@fill_doc
def plot_compare_evokeds(evokeds, picks=None, colors=None,
                         linestyles=None, styles=None, cmap=None,
                         vlines='auto', ci=True, truncate_yaxis='auto',
                         truncate_xaxis=True, ylim=None, invert_y=False,
                         show_sensors=None, legend=True,
                         split_legend=None, axes=None, title=None, show=True,
                         combine=None, sphere=None):
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

    colors : list | dict | None
        Colors to use when plotting the ERP/F lines and confidence bands. If
        ``cmap`` is not ``None``, ``colors`` must be a :class:`list` or
        :class:`dict` of :class:`ints <int>` or :class:`floats <float>`
        indicating steps or percentiles (respectively) along the colormap. If
        ``cmap`` is ``None``, list elements or dict values of ``colors`` must
        be :class:`ints <int>` or valid :doc:`matplotlib colors
        <tutorials/colors/colors>`; lists are cycled through sequentially,
        while dicts must have keys matching the keys or conditions of an
        ``evokeds`` dict (see Notes for details). If ``None``, the current
        :doc:`matplotlib color cycle <gallery/color/color_cycle_default>` is
        used. Defaults to ``None``.
    linestyles : list | dict | None
        Styles to use when plotting the ERP/F lines. If a :class:`list` or
        :class:`dict`, elements must be valid :doc:`matplotlib linestyles
        <matplotlib:gallery/lines_bars_and_markers/linestyles>`. Lists are
        cycled through sequentially; dictionaries must have keys matching the
        keys or conditions of an ``evokeds`` dict (see Notes for details). If
        ``None``, all lines will be solid. Defaults to ``None``.
    styles : dict | None
        Dictionary of styles to use when plotting ERP/F lines. Keys must match
        keys or conditions of ``evokeds``, and values must be a :class:`dict`
        of legal inputs to :func:`matplotlib.pyplot.plot`. Those values will be
        passed as parameters to the line plot call of the corresponding
        condition, overriding defaults (e.g.,
        ``styles={"Aud/L": {"linewidth": 3}}`` will set the linewidth for
        "Aud/L" to 3). As with ``colors`` and ``linestyles``, keys matching
        conditions in ``/``-separated ``evokeds`` keys are supported (see Notes
        for details).
    cmap : None | str | tuple | instance of matplotlib.colors.Colormap
        Colormap from which to draw color values when plotting the ERP/F lines
        and confidence bands. If not ``None``, ints or floats in the ``colors``
        parameter are mapped to steps or percentiles (respectively) along the
        colormap. If ``cmap`` is a :class:`str`, it will be passed to
        :func:`matplotlib.cm.get_cmap`; if ``cmap`` is a tuple, its first
        element will be used as a string to label the colorbar, and its
        second element will be passed to :func:`matplotlib.cm.get_cmap` (unless
        it is already an instance of :class:`~matplotlib.colors.Colormap`).

        .. versionchanged:: 0.19
            Support for passing :class:`~matplotlib.colors.Colormap` instances.

    vlines : "auto" | list of float
        A list in seconds at which to plot dashed vertical lines.
        If "auto" and the supplied data includes 0, it is set to [0.]
        and a vertical bar is plotted at time 0. If an empty list is passed,
        no vertical lines are plotted.
    ci : float | bool | callable | None
        Confidence band around each ERP/F time series. If ``False`` or ``None``
        no confidence band is drawn. If :class:`float`, ``ci`` must be between
        0 and 1, and will set the threshold for a bootstrap
        (single plot)/parametric (when ``axes=='topo'``)  estimation of the
        confidence band; ``True`` is equivalent to setting a threshold of 0.95
        (i.e., the 95%% confidence band is drawn). If a callable, it must take
        a single array (n_observations × n_times) as input and return upper and
        lower confidence margins (2 × n_times). Defaults to ``True``.
    truncate_yaxis : bool | 'auto'
        Whether to shorten the y-axis spine. If 'auto', the spine is truncated
        at the minimum and maximum ticks. If ``True``, it is truncated at the
        multiple of 0.25 nearest to half the maximum absolute value of the
        data. If ``truncate_xaxis=False``, only the far bound of the y-axis
        will be truncated. Defaults to 'auto'.
    truncate_xaxis : bool
        Whether to shorten the x-axis spine. If ``True``, the spine is
        truncated at the minimum and maximum ticks. If
        ``truncate_yaxis=False``, only the far bound of the x-axis will be
        truncated. Defaults to ``True``.
    ylim : dict | None
        Y-axis limits for plots (after scaling has been applied). :class:`dict`
        keys should match channel types; valid keys are eeg, mag, grad, misc
        (example: ``ylim=dict(eeg=[-20, 20])``). If ``None``, the y-axis limits
        will be set automatically by matplotlib. Defaults to ``None``.
    invert_y : bool
        Whether to plot negative values upward (as is sometimes done
        for ERPs out of tradition). Defaults to ``False``.
    show_sensors : bool | int | str | None
        Whether to display an inset showing sensor locations on a head outline.
        If :class:`int` or :class:`str`, indicates position of the inset (see
        :func:`mpl_toolkits.axes_grid1.inset_locator.inset_axes`). If ``None``,
        treated as ``True`` if there is only one channel in ``picks``. If
        ``True``, location is upper or lower right corner, depending on data
        values. Defaults to ``None``.
    legend : bool | int | str
        Whether to show a legend for the colors/linestyles of the conditions
        plotted. If :class:`int` or :class:`str`, indicates position of the
        legend (see :func:`mpl_toolkits.axes_grid1.inset_locator.inset_axes`).
        If ``True``, equivalent to ``'upper left'``. Defaults to ``True``.
    split_legend : bool | None
        Whether to separate color and linestyle in the legend. If ``None``,
        a separate linestyle legend will still be shown if ``cmap`` is
        specified. Defaults to ``None``.
    axes : None | Axes instance | list of Axes | 'topo'
        :class:`~matplotlib.axes.Axes` object to plot into. If plotting
        multiple channel types (or multiple channels when ``combine=None``),
        ``axes`` should be a list of appropriate length containing
        :class:`~matplotlib.axes.Axes` objects. If ``'topo'``, a new
        :class:`~matplotlib.figure.Figure` is created with one axis for each
        channel, in a topographical layout. If ``None``, a new
        :class:`~matplotlib.figure.Figure` is created for each channel type.
        Defaults to ``None``.
    title : str | None
        Title printed above the plot. If ``None``, a title will be
        automatically generated based on channel name(s) or type(s) and the
        value of the ``combine`` parameter. Defaults to ``None``.
    show : bool
        Whether to show the figure. Defaults to ``True``.
    %(combine)s
        If callable, the callable must accept one positional input (data of
        shape ``(n_evokeds, n_channels, n_times)``) and return an
        :class:`array <numpy.ndarray>` of shape ``(n_epochs, n_times)``. For
        example::

            combine = lambda data: np.median(data, axis=1)

        If ``combine`` is ``None``, channels are combined by computing GFP,
        unless ``picks`` is a single channel (not channel type) or
        ``axes='topo'``, in which cases no combining is performed. Defaults to
        ``None``.
    %(topomap_sphere_auto)s

    Returns
    -------
    fig : list of Figure instances
        A list of the figure(s) generated.

    Notes
    -----
    If the parameters ``styles``, ``colors``, or ``linestyles`` are passed as
    :class:`dicts <dict>`, then ``evokeds`` must also be a :class:`dict`, and
    the keys of the plot-style parameters must either match the keys of
    ``evokeds``, or match a ``/``-separated partial key ("condition") of
    ``evokeds``. For example, if evokeds has keys "Aud/L", "Aud/R", "Vis/L",
    and "Vis/R", then ``linestyles=dict(L='--', R='-')`` will plot both Aud/L
    and Vis/L conditions with dashed lines and both Aud/R and Vis/R conditions
    with solid lines. Similarly, ``colors=dict(Aud='r', Vis='b')`` will plot
    Aud/L and Aud/R conditions red and Vis/L and Vis/R conditions blue.

    Color specification depends on whether a colormap has been provided in the
    ``cmap`` parameter. The following table summarizes how the ``colors``
    parameter is interpreted:

    .. cssclass:: table-bordered
    .. rst-class:: midvalign

    +-------------+----------------+------------------------------------------+
    | ``cmap``    | ``colors``     | result                                   |
    +=============+================+==========================================+
    |             | None           | matplotlib default color cycle; unique   |
    |             |                | color for each condition                 |
    |             +----------------+------------------------------------------+
    |             |                | matplotlib default color cycle; lowest   |
    |             | list or dict   | integer mapped to first cycle color;     |
    |             | of integers    | conditions with same integer get same    |
    | None        |                | color; unspecified conditions are "gray" |
    |             +----------------+------------------------------------------+
    |             | list or dict   | ``ValueError``                           |
    |             | of floats      |                                          |
    |             +----------------+------------------------------------------+
    |             | list or dict   | the specified hex colors; unspecified    |
    |             | of hexadecimal | conditions are "gray"                    |
    |             | color strings  |                                          |
    +-------------+----------------+------------------------------------------+
    |             | None           | equally spaced colors on the colormap;   |
    |             |                | unique color for each condition          |
    |             +----------------+------------------------------------------+
    |             |                | equally spaced colors on the colormap;   |
    |             | list or dict   | lowest integer mapped to first cycle     |
    | string or   | of integers    | color; conditions with same integer      |
    | instance of |                | get same color                           |
    | matplotlib  +----------------+------------------------------------------+
    | Colormap    | list or dict   | floats mapped to corresponding colormap  |
    |             | of floats      | values                                   |
    |             +----------------+------------------------------------------+
    |             | list or dict   |                                          |
    |             | of hexadecimal | ``TypeError``                            |
    |             | color strings  |                                          |
    +-------------+----------------+------------------------------------------+
    """
    import matplotlib.pyplot as plt
    from ..evoked import Evoked, _check_evokeds_ch_names_times

    # build up evokeds into a dict, if it's not already
    if isinstance(evokeds, Evoked):
        evokeds = [evokeds]
    if isinstance(evokeds, (list, tuple)):
        evokeds = {str(idx + 1): evk for idx, evk in enumerate(evokeds)}
    if not isinstance(evokeds, dict):
        raise TypeError('"evokeds" must be a dict, list, or instance of '
                        'mne.Evoked; got {}'.format(type(evokeds).__name__))
    evokeds = deepcopy(evokeds)  # avoid modifying dict outside function scope
    for cond, evoked in evokeds.items():
        _validate_type(cond, 'str', 'Conditions')
        if isinstance(evoked, Evoked):
            evokeds[cond] = [evoked]  # wrap singleton evokeds in a list
        for evk in evokeds[cond]:
            _validate_type(evk, Evoked, 'All evokeds entries ', 'Evoked')
    # ensure same channels and times across all evokeds
    all_evoked = sum(evokeds.values(), [])
    _check_evokeds_ch_names_times(all_evoked)
    del all_evoked

    # get some representative info
    conditions = list(evokeds)
    one_evoked = evokeds[conditions[0]][0]
    times = one_evoked.times
    info = one_evoked.info
    sphere = _check_sphere(sphere, info)
    tmin, tmax = times[0], times[-1]
    # set some defaults
    if ylim is None:
        ylim = dict()
    if vlines == 'auto':
        vlines = [0.] if (tmin < 0 < tmax) else []
    _validate_type(vlines, (list, tuple), 'vlines', 'list or tuple')

    # is picks a channel type (or None)?
    orig_picks = deepcopy(picks)
    picks, picked_types = _picks_to_idx(info, picks, return_kind=True)
    # some things that depend on picks:
    ch_names = np.array(one_evoked.ch_names)[picks].tolist()
    ch_types = list(_get_channel_types(info, picks=picks, unique=True)
                    .intersection(_DATA_CH_TYPES_SPLIT + ('misc',)))  # miscICA
    picks_by_type = channel_indices_by_type(info, picks)
    # discard picks from non-data channels (e.g., ref_meg)
    good_picks = sum([picks_by_type[ch_type] for ch_type in ch_types], [])
    picks = np.intersect1d(picks, good_picks)
    if show_sensors is None:
        show_sensors = (len(picks) == 1)

    # cannot combine a single channel
    if (len(picks) < 2) and combine is not None:
        warn('Only {} channel in "picks"; cannot combine by method "{}".'
             .format(len(picks), combine))
    # `combine` defaults to GFP unless picked a single channel or axes='topo'
    if combine is None and len(picks) > 1 and axes != 'topo':
        combine = 'gfp'
    # convert `combine` into callable (if None or str)
    combine_func = _make_combine_callable(combine)

    # title
    title = _title_helper_pce(title, picked_types, picks=orig_picks,
                              ch_names=ch_names, combine=combine)

    # setup axes
    do_topo = (axes == 'topo')
    if do_topo:
        show_sensors = False
        if len(picks) > 70:
            logger.info('You are plotting to a topographical layout with >70 '
                        'sensors. This can be extremely slow. Consider using '
                        'mne.viz.plot_topo, which is optimized for speed.')
        axes = ['topo'] * len(ch_types)
    else:
        if axes is None:
            axes = (plt.subplots(figsize=(8, 6))[1] for _ in ch_types)
        elif isinstance(axes, plt.Axes):
            axes = [axes]
            _validate_if_list_of_axes(axes, obligatory_len=len(ch_types))

    if len(ch_types) > 1:
        logger.info('Multiple channel types selected, returning one figure '
                    'per type.')
        figs = list()
        for ch_type, ax in zip(ch_types, axes):
            _picks = picks_by_type[ch_type]
            _ch_names = np.array(one_evoked.ch_names)[_picks].tolist()
            _picks = ch_type if picked_types else _picks
            # don't pass `combine` here; title will run through this helper
            # function a second time & it will get added then
            _title = _title_helper_pce(title, picked_types, picks=_picks,
                                       ch_names=_ch_names, combine=None)
            figs.extend(plot_compare_evokeds(
                evokeds, picks=_picks, colors=colors, cmap=cmap,
                linestyles=linestyles, styles=styles, vlines=vlines, ci=ci,
                truncate_yaxis=truncate_yaxis, ylim=ylim, invert_y=invert_y,
                legend=legend, show_sensors=show_sensors,
                axes=ax, title=_title, split_legend=split_legend, show=show,
                sphere=sphere))
        return figs

    # colors and colormap. This yields a `styles` dict with one entry per
    # condition, specifying at least color and linestyle. THIS MUST BE DONE
    # AFTER THE "MULTIPLE CHANNEL TYPES" LOOP
    (_styles, _linestyles, _colors, _cmap, colorbar_title,
     colorbar_ticks) = _handle_styles_pce(styles, linestyles, colors, cmap,
                                          conditions)
    # From now on there is only 1 channel type
    assert len(ch_types) == 1
    ch_type = ch_types[0]
    # some things that depend on ch_type:
    units = _handle_default('units')[ch_type]
    scalings = _handle_default('scalings')[ch_type]

    # prep for topo
    pos_picks = picks  # need this version of picks for sensor location inset
    info = pick_info(info, sel=picks, copy=True)
    all_ch_names = info['ch_names']
    if not do_topo:
        # add vacuous "index" (needed for topo) so same code works for both
        axes = [(ax, 0) for ax in axes]
        if np.array(picks).ndim < 2:
            picks = [picks]  # enables zipping w/ axes
    else:
        from .topo import iter_topography
        fig = plt.figure(figsize=(18, 14))

        def click_func(
                ax_, pick_, evokeds=evokeds, colors=colors,
                linestyles=linestyles, styles=styles, cmap=cmap, vlines=vlines,
                ci=ci, truncate_yaxis=truncate_yaxis,
                truncate_xaxis=truncate_xaxis, ylim=ylim, invert_y=invert_y,
                show_sensors=show_sensors, legend=legend,
                split_legend=split_legend, picks=picks, combine=combine):
            plot_compare_evokeds(
                evokeds=evokeds, colors=colors, linestyles=linestyles,
                styles=styles, cmap=cmap, vlines=vlines, ci=ci,
                truncate_yaxis=truncate_yaxis, truncate_xaxis=truncate_xaxis,
                ylim=ylim, invert_y=invert_y, show_sensors=show_sensors,
                legend=legend, split_legend=split_legend,
                picks=picks[pick_], combine=combine, axes=ax_, show=True,
                sphere=sphere)

        layout = find_layout(info)
        # shift everything to the right by 15% of one axes width
        layout.pos[:, 0] += layout.pos[0, 2] * .15
        layout.pos[:, 1] += layout.pos[0, 3] * .15
        # `axes` will be a list of (axis_object, channel_index) tuples
        axes = list(iter_topography(
            info, layout=layout, on_pick=click_func,
            fig=fig, fig_facecolor='w', axis_facecolor='w',
            axis_spinecolor='k', layout_scale=.925, legend=True))
        picks = list(picks)
    del info

    # for each axis, compute the grand average and (maybe) the CI
    # (per sensor if topo, otherwise aggregating over sensors)
    c_func = None if do_topo else combine_func
    all_data = list()
    all_cis = list()
    for _picks, (ax, idx) in zip(picks, axes):
        data_dict = dict()
        ci_dict = dict()
        for cond in conditions:
            this_evokeds = evokeds[cond]
            # skip CIs when possible; assign ci_fun first to get arg checking
            ci_fun = _get_ci_function_pce(ci, do_topo=do_topo)
            ci_fun = ci_fun if len(this_evokeds) > 1 else None
            res = _get_data_and_ci(this_evokeds, combine, c_func, picks=_picks,
                                   scaling=scalings, ci_fun=ci_fun)
            data_dict[cond] = res[0]
            if ci_fun is not None:
                ci_dict[cond] = res[1]
        all_data.append(data_dict)  # grand means, or indiv. sensors if do_topo
        all_cis.append(ci_dict)
    del evokeds

    # compute ylims
    allvalues = list()
    for _dict in all_data:
        for _array in list(_dict.values()):
            allvalues.append(_array[np.newaxis])  # to get same .ndim as CIs
    for _dict in all_cis:
        allvalues.extend(list(_dict.values()))
    allvalues = np.concatenate(allvalues)
    norm = np.all(allvalues > 0)
    orig_ymin, orig_ymax = ylim.get(ch_type, [None, None])
    ymin, ymax = _setup_vmin_vmax(allvalues, orig_ymin, orig_ymax, norm)
    del allvalues

    # add empty data and title for the legend axis
    if do_topo:
        all_data.append({cond: np.array([]) for cond in data_dict})
        all_cis.append({cond: None for cond in ci_dict})
        all_ch_names.append('')

    # plot!
    for (ax, idx), data, cis in zip(axes, all_data, all_cis):
        if do_topo:
            title = all_ch_names[idx]
        # plot the data
        _times = [] if idx == -1 else times
        _plot_compare_evokeds(ax, data, conditions, _times, cis, _styles,
                              title, norm, do_topo)
        # draw axes & vlines
        skip_axlabel = do_topo and (idx != -1)
        _draw_axes_pce(ax, ymin, ymax, truncate_yaxis, truncate_xaxis,
                       invert_y, vlines, tmin, tmax, units, skip_axlabel)
    # add inset scalp plot showing location of sensors picked
    if show_sensors:
        _validate_type(show_sensors, (np.int, bool, str, type(None)),
                       'show_sensors', 'numeric, str, None or bool')
        if not _check_ch_locs(np.array(one_evoked.info['chs'])[pos_picks]):
            warn('Cannot find channel coordinates in the supplied Evokeds. '
                 'Not showing channel locations.')
        else:
            _evoked_sensor_legend(one_evoked.info, pos_picks, ymin, ymax,
                                  show_sensors, ax, sphere)
    # add color/linestyle/colormap legend(s)
    if legend:
        _draw_legend_pce(legend, split_legend, _styles, _linestyles, _colors,
                         _cmap, do_topo, ax)
    if cmap is not None:
        _draw_colorbar_pce(ax, _colors, _cmap, colorbar_title, colorbar_ticks)
    # finish
    plt_show(show)
    return [ax.figure]
