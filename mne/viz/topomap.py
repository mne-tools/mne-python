"""Functions to plot M/EEG data e.g. topographies."""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import math
import copy
from functools import partial

import numpy as np
from scipy import linalg

from ..baseline import rescale
from ..io.constants import FIFF
from ..io.pick import (pick_types, _picks_by_type, channel_type, pick_info,
                       _pick_data_channels)
from ..utils import _clean_names, _time_mask, verbose, logger, warn
from .utils import (tight_layout, _setup_vmin_vmax, _prepare_trellis,
                    _check_delayed_ssp, _draw_proj_checkbox, figure_nobar,
                    plt_show, _process_times, DraggableColorbar,
                    _validate_if_list_of_axes, _setup_cmap)
from ..time_frequency import psd_multitaper
from ..defaults import _handle_default
from ..channels.layout import _find_topomap_coords
from ..io.meas_info import Info


def _prepare_topo_plot(inst, ch_type, layout):
    """Prepare topo plot."""
    info = copy.deepcopy(inst if isinstance(inst, Info) else inst.info)

    if layout is None and ch_type is not 'eeg':
        from ..channels import find_layout
        layout = find_layout(info)
    elif layout == 'auto':
        layout = None

    clean_ch_names = _clean_names(info['ch_names'])
    for ii, this_ch in enumerate(info['chs']):
        this_ch['ch_name'] = clean_ch_names[ii]
    info._update_redundant()
    info._check_consistency()

    # special case for merging grad channels
    if (ch_type == 'grad' and FIFF.FIFFV_COIL_VV_PLANAR_T1 in
            np.unique([ch['coil_type'] for ch in info['chs']])):
        from ..channels.layout import _pair_grad_sensors
        picks, pos = _pair_grad_sensors(info, layout)
        merge_grads = True
    else:
        merge_grads = False
        if ch_type == 'eeg':
            picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                               exclude='bads')
        else:
            picks = pick_types(info, meg=ch_type, ref_meg=False,
                               exclude='bads')

        if len(picks) == 0:
            raise ValueError("No channels of type %r" % ch_type)

        if layout is None:
            pos = _find_topomap_coords(info, picks)
        else:
            names = [n.upper() for n in layout.names]
            pos = list()
            for pick in picks:
                this_name = info['ch_names'][pick].upper()
                if this_name in names:
                    pos.append(layout.pos[names.index(this_name)])
                else:
                    warn('Failed to locate %s channel positions from layout. '
                         'Inferring channel positions from data.' % ch_type)
                    pos = _find_topomap_coords(info, picks)
                    break

    ch_names = [info['ch_names'][k] for k in picks]
    if merge_grads:
        # change names so that vectorview combined grads appear as MEG014x
        # instead of MEG0142 or MEG0143 which are the 2 planar grads.
        ch_names = [ch_names[k][:-1] + 'x' for k in range(0, len(ch_names), 2)]
    pos = np.array(pos)[:, :2]  # 2D plot, otherwise interpolation bugs
    return picks, pos, merge_grads, ch_names, ch_type


def _plot_update_evoked_topomap(params, bools):
    """Update topomaps."""
    projs = [proj for ii, proj in enumerate(params['projs'])
             if ii in np.where(bools)[0]]

    params['proj_bools'] = bools
    new_evoked = params['evoked'].copy()
    new_evoked.info['projs'] = []
    new_evoked.add_proj(projs)
    new_evoked.apply_proj()

    data = new_evoked.data[np.ix_(params['picks'],
                                  params['time_idx'])] * params['scale']
    if params['merge_grads']:
        from ..channels.layout import _merge_grad_data
        data = _merge_grad_data(data)
    image_mask = params['image_mask']

    pos_x, pos_y = np.asarray(params['pos'])[:, :2].T

    xi = np.linspace(pos_x.min(), pos_x.max(), params['res'])
    yi = np.linspace(pos_y.min(), pos_y.max(), params['res'])
    Xi, Yi = np.meshgrid(xi, yi)
    for ii, im in enumerate(params['images']):
        Zi = _griddata(pos_x, pos_y, data[:, ii], Xi, Yi)
        Zi[~image_mask] = np.nan
        im.set_data(Zi)
    for cont in params['contours']:
        cont.set_array(np.c_[Xi, Yi, Zi])

    params['fig'].canvas.draw()


def plot_projs_topomap(projs, layout=None, cmap=None, sensors=True,
                       colorbar=False, res=64, size=1, show=True,
                       outlines='head', contours=6, image_interp='bilinear',
                       axes=None):
    """Plot topographic maps of SSP projections.

    Parameters
    ----------
    projs : list of Projection
        The projections
    layout : None | Layout | list of Layout
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). Or a list of Layout if projections
        are from different sensor types.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode (only works if ``colorbar=True``) the colors are
        adjustable by clicking and dragging the colorbar with left and right
        mouse button. Left mouse button moves the scale up and down and right
        mouse button adjusts the range. Hitting space bar resets the range. Up
        and down arrows can be used to change the colormap. If None (default),
        'Reds' is used for all positive data, otherwise defaults to 'RdBu_r'.
        If 'interactive', translates to (None, True).
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    colorbar : bool
        Plot a colorbar.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : scalar
        Side length of the topomaps in inches (only applies when plotting
        multiple topomaps at a time).
    show : bool
        Show figure if True.
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    axes : instance of Axes | list | None
        The axes to plot to. If list, the list must be a list of Axes of
        the same length as the number of projectors. If instance of Axes,
        there must be only one projector. Defaults to None.

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if layout is None:
        from ..channels import read_layout
        layout = read_layout('Vectorview-all')

    if not isinstance(layout, list):
        layout = [layout]

    n_projs = len(projs)
    nrows = math.floor(math.sqrt(n_projs))
    ncols = math.ceil(n_projs / nrows)

    cmap = _setup_cmap(cmap)
    if axes is None:
        plt.figure()
        axes = list()
        for idx in range(len(projs)):
            ax = plt.subplot(nrows, ncols, idx + 1)
            axes.append(ax)
    elif isinstance(axes, plt.Axes):
        axes = [axes]
    if len(axes) != len(projs):
        raise RuntimeError('There must be an axes for each picked projector.')
    for proj_idx, proj in enumerate(projs):
        axes[proj_idx].set_title(proj['desc'][:10] + '...')
        ch_names = _clean_names(proj['data']['col_names'],
                                remove_whitespace=True)
        data = proj['data']['data'].ravel()

        idx = []
        for l in layout:
            is_vv = l.kind.startswith('Vectorview')
            if is_vv:
                from ..channels.layout import _pair_grad_sensors_from_ch_names
                grad_pairs = _pair_grad_sensors_from_ch_names(ch_names)
                if grad_pairs:
                    ch_names = [ch_names[i] for i in grad_pairs]

            l_names = _clean_names(l.names, remove_whitespace=True)
            idx = [l_names.index(c) for c in ch_names if c in l_names]
            if len(idx) == 0:
                continue

            pos = l.pos[idx]
            if is_vv and grad_pairs:
                from ..channels.layout import _merge_grad_data
                shape = (len(idx) // 2, 2, -1)
                pos = pos.reshape(shape).mean(axis=1)
                data = _merge_grad_data(data[grad_pairs]).ravel()

            break

        if len(idx):
            im = plot_topomap(data, pos[:, :2], vmax=None, cmap=cmap[0],
                              sensors=sensors, res=res, axes=axes[proj_idx],
                              outlines=outlines, contours=contours,
                              image_interp=image_interp, show=False)[0]
            if colorbar:
                divider = make_axes_locatable(axes[proj_idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, cmap=cmap)
                if cmap[1]:
                    axes[proj_idx].CB = DraggableColorbar(cbar, im)
        else:
            raise RuntimeError('Cannot find a proper layout for projection %s'
                               % proj['desc'])
    tight_layout(fig=axes[0].get_figure())
    plt_show(show)
    return axes[0].get_figure()


def _check_outlines(pos, outlines, head_pos=None):
    """Check or create outlines for topoplot."""
    pos = np.array(pos, float)[:, :2]  # ensure we have a copy
    head_pos = dict() if head_pos is None else head_pos
    if not isinstance(head_pos, dict):
        raise TypeError('head_pos must be dict or None')
    head_pos = copy.deepcopy(head_pos)
    for key in head_pos.keys():
        if key not in ('center', 'scale'):
            raise KeyError('head_pos must only contain "center" and '
                           '"scale"')
        head_pos[key] = np.array(head_pos[key], float)
        if head_pos[key].shape != (2,):
            raise ValueError('head_pos["%s"] must have shape (2,), not '
                             '%s' % (key, head_pos[key].shape))

    if outlines in ('head', 'skirt', None):
        radius = 0.5
        l = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(l) * radius
        head_y = np.sin(l) * radius
        nose_x = np.array([0.18, 0, -0.18]) * radius
        nose_y = np.array([radius - .004, radius * 1.15, radius - .004])
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                         .532, .510, .489])
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199])

        # shift and scale the electrode positions
        if 'center' not in head_pos:
            head_pos['center'] = 0.5 * (pos.max(axis=0) + pos.min(axis=0))
        pos -= head_pos['center']

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x, ear_y),
                                 ear_right=(-ear_x, ear_y))
        else:
            outlines_dict = dict()

        if outlines == 'skirt':
            if 'scale' not in head_pos:
                # By default, fit electrodes inside the head circle
                head_pos['scale'] = 1.0 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']

            # Make the figure encompass slightly more than all points
            mask_scale = 1.25 * (pos.max(axis=0) - pos.min(axis=0))

            outlines_dict['autoshrink'] = False
            outlines_dict['mask_pos'] = (mask_scale[0] * head_x,
                                         mask_scale[1] * head_y)
            outlines_dict['clip_radius'] = (mask_scale / 2.)
        else:
            if 'scale' not in head_pos:
                # The default is to make the points occupy a slightly smaller
                # proportion (0.85) of the total width and height
                # this number was empirically determined (seems to work well)
                head_pos['scale'] = 0.85 / (pos.max(axis=0) - pos.min(axis=0))
            pos *= head_pos['scale']
            outlines_dict['autoshrink'] = True
            outlines_dict['mask_pos'] = head_x, head_y
            outlines_dict['clip_radius'] = (0.5, 0.5)

        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image'
                             'mask')
    else:
        raise ValueError('Invalid value for `outlines')

    return pos, outlines


def _draw_outlines(ax, outlines):
    """Draw the outlines for a topomap."""
    outlines_ = dict([(k, v) for k, v in outlines.items() if k not in
                      ['patch', 'autoshrink']])
    for key, (x_coord, y_coord) in outlines_.items():
        if 'mask' in key:
            continue
        ax.plot(x_coord, y_coord, color='k', linewidth=1, clip_on=False)
    return outlines_


def _griddata(x, y, v, xi, yi):
    """Make griddata."""
    xy = x.ravel() + y.ravel() * -1j
    d = xy[None, :] * np.ones((len(xy), 1))
    d = np.abs(d - d.T)
    n = d.shape[0]
    d.flat[::n + 1] = 1.

    g = (d * d) * (np.log(d) - 1.)
    g.flat[::n + 1] = 0.
    weights = linalg.solve(g, v.ravel())

    m, n = xi.shape
    zi = np.zeros_like(xi)
    xy = xy.T

    g = np.empty(xy.shape)
    for i in range(m):
        for j in range(n):
            d = np.abs(xi[i, j] + -1j * yi[i, j] - xy)
            mask = np.where(d == 0)[0]
            if len(mask):
                d[mask] = 1.
            np.log(d, out=g)
            g -= 1.
            g *= d * d
            if len(mask):
                g[mask] = 0.
            zi[i, j] = g.dot(weights)
    return zi


def _plot_sensors(pos_x, pos_y, sensors, ax):
    """Plot sensors."""
    from matplotlib.patches import Circle
    if sensors is True:
        for x, y in zip(pos_x, pos_y):
            ax.add_artist(Circle(xy=(x, y), radius=0.003, color='k'))
    else:
        ax.plot(pos_x, pos_y, sensors)


def plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', image_mask=None,
                 contours=6, image_interp='bilinear', show=True,
                 head_pos=None, onselect=None):
    """Plot a topographic map as image.

    Parameters
    ----------
    data : array, shape (n_chan,)
        The data values to plot.
    pos : array, shape (n_chan, 2) | instance of Info
        Location information for the data points(/channels).
        If an array, for each data point, the x and y coordinates.
        If an Info object, it must contain only one data type and
        exactly `len(data)` data channels, and the x/y coordinates will
        be inferred from this Info object.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | None
        Colormap to use. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    res : int
        The resolution of the topomap image (n pixels along each side).
    axes : instance of Axes | None
        The axes to plot to. If None, the current axes will be used.
    names : list | None
        List of channel names. If None, channel names are not plotted.
    show_names : bool | callable
        If True, show channel names on top of the map. If a callable is
        passed, channel names will be formatted using the callable; e.g., to
        delete the prefix 'MEG ' from all channel names, pass the function
        lambda x: x.replace('MEG ', ''). If `mask` is not None, only
        significant sensors will be shown.
        If `True`, a list of names must be provided (see `names` keyword).
    mask : ndarray of bool, shape (n_channels, n_times) | None
        The channels to be marked as significant at a given time point.
        Indices set to `True` will be considered. Defaults to None.
    mask_params : dict | None
        Additional plotting parameters for plotting significant sensors.
        Default (None) equals::

           dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)

    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axes plots). If None, nothing will be drawn.
        Defaults to 'head'.
    image_mask : ndarray of bool, shape (res, res) | None
        The image mask to cover the interpolated surface. If None, it will be
        computed from the outline.
    contours : int | False | array of float | None
        The number of contour lines to draw. If 0, no contours will be drawn.
        If an array, the values represent the levels for the contours. The
        values are in uV for EEG, fT for magnetometers and fT/m for
        gradiometers.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    show : bool
        Show figure if True.
    head_pos : dict | None
        If None (default), the sensors are positioned such that they span
        the head circle. If dict, can have entries 'center' (tuple) and
        'scale' (tuple) for what the center and scale of the head should be
        relative to the electrode locations.
    onselect : callable | None
        Handle for a function that is called when the user selects a set of
        channels by rectangle selection (matplotlib ``RectangleSelector``). If
        None interactive selection is disabled. Defaults to None.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    data = np.asarray(data)
    logger.debug('Plotting topomap for data shape %s' % (data.shape,))

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = set(channel_type(pos, idx)
                      for idx, _ in enumerate(pos["chs"]))
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.channels.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object and "
                             "the data array does not match. " + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            from ..channels.layout import (_merge_grad_data, find_layout,
                                           _pair_grad_sensors)
            picks, pos = _pair_grad_sensors(pos, find_layout(pos))
            data = _merge_grad_data(data[picks]).reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks)

    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    pos, outlines = _check_outlines(pos, outlines, head_pos)

    ax = axes if axes else plt.gca()
    pos_x, pos_y = _prepare_topomap(pos, ax)
    if outlines is None:
        xmin, xmax = pos_x.min(), pos_x.max()
        ymin, ymax = pos_y.min(), pos_y.max()
    else:
        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        mask_ = np.c_[outlines['mask_pos']]
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                      np.max(np.r_[xlim[1], mask_[:, 0]]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                      np.max(np.r_[ylim[1], mask_[:, 1]]))

    # interpolate data
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _griddata(pos_x, pos_y, data, Xi, Yi)

    if outlines is None:
        _is_default_outlines = False
    elif isinstance(outlines, dict):
        _is_default_outlines = any(k.startswith('head') for k in outlines)

    if _is_default_outlines and image_mask is None:
        # prepare masking
        image_mask, pos = _make_image_mask(outlines, pos, res)

    mask_params = _handle_default('mask_params', mask_params)

    # plot outline
    linewidth = mask_params['markeredgewidth']
    patch = None
    if 'patch' in outlines:
        patch = outlines['patch']
        patch_ = patch() if callable(patch) else patch
        patch_.set_clip_on(False)
        ax.add_patch(patch_)
        ax.set_transform(ax.transAxes)
        ax.set_clip_path(patch_)

    # plot map and countour
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax),
                   interpolation=image_interp)

    # This tackles an incomprehensible matplotlib bug if no contours are
    # drawn. To avoid rescalings, we will always draw contours.
    # But if no contours are desired we only draw one and make it invisible .
    no_contours = False
    if isinstance(contours, (np.ndarray, list)):
        pass  # contours precomputed
    elif contours in (False, None):
        contours, no_contours = 1, True
    cont = ax.contour(Xi, Yi, Zi, contours, colors='k',
                      linewidths=linewidth)
    if no_contours is True:
        for col in cont.collections:
            col.set_visible(False)

    if _is_default_outlines:
        from matplotlib import patches
        patch_ = patches.Ellipse((0, 0),
                                 2 * outlines['clip_radius'][0],
                                 2 * outlines['clip_radius'][1],
                                 clip_on=True,
                                 transform=ax.transData)
    if _is_default_outlines or patch is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    if sensors is not False and mask is None:
        _plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(ax, outlines)

    if show_names:
        if names is None:
            raise ValueError("To show names, a list of names must be provided"
                             " (see `names` keyword).")
        if show_names is True:
            def _show_names(x):
                return x
        else:
            _show_names = show_names
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = _show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', size='x-small')

    plt.subplots_adjust(top=.95)

    if onselect is not None:
        ax.RS = RectangleSelector(ax, onselect=onselect)
    plt_show(show)
    return im, cont


def _make_image_mask(outlines, pos, res):
    """Make an image mask."""
    mask_ = np.c_[outlines['mask_pos']]
    xmin, xmax = (np.min(np.r_[np.inf, mask_[:, 0]]),
                  np.max(np.r_[-np.inf, mask_[:, 0]]))
    ymin, ymax = (np.min(np.r_[np.inf, mask_[:, 1]]),
                  np.max(np.r_[-np.inf, mask_[:, 1]]))

    if outlines.get('autoshrink', False) is not False:
        inside = _inside_contour(pos, mask_)
        outside = np.invert(inside)
        outlier_points = pos[outside]
        while np.any(outlier_points):  # auto shrink
            pos *= 0.99
            inside = _inside_contour(pos, mask_)
            outside = np.invert(inside)
            outlier_points = pos[outside]

    image_mask = np.zeros((res, res), dtype=bool)
    xi_mask = np.linspace(xmin, xmax, res)
    yi_mask = np.linspace(ymin, ymax, res)
    Xi_mask, Yi_mask = np.meshgrid(xi_mask, yi_mask)

    pos_ = np.c_[Xi_mask.flatten(), Yi_mask.flatten()]
    inds = _inside_contour(pos_, mask_)
    image_mask[inds.reshape(image_mask.shape)] = True

    return image_mask, pos


def _inside_contour(pos, contour):
    """Check if points are inside a contour."""
    npos = len(pos)
    x, y = pos[:, :2].T

    check_mask = np.ones((npos), dtype=bool)
    check_mask[((x < np.min(x)) | (y < np.min(y)) |
                (x > np.max(x)) | (y > np.max(y)))] = False

    critval = 0.1
    sel = np.where(check_mask)[0]
    for this_sel in sel:
        contourx = contour[:, 0] - pos[this_sel, 0]
        contoury = contour[:, 1] - pos[this_sel, 1]
        angle = np.arctan2(contoury, contourx)
        angle = np.unwrap(angle)
        total = np.sum(np.diff(angle))
        check_mask[this_sel] = np.abs(total) > critval

    return check_mask


def _plot_ica_topomap(ica, idx=0, ch_type=None, res=64, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', colorbar=False,
                      title=None, show=True, outlines='head', contours=6,
                      image_interp='bilinear', head_pos=None, axes=None):
    """Plot single ica map to axes."""
    import matplotlib as mpl
    from ..channels import _get_ch_type

    if ica.info is None:
        raise RuntimeError('The ICA\'s measurement info is missing. Please '
                           'fit the ICA or add the corresponding info object.')
    if not isinstance(axes, mpl.axes.Axes):
        raise ValueError('axis has to be an instance of matplotlib Axes, '
                         'got %s instead.' % type(axes))
    ch_type = _get_ch_type(ica, ch_type)

    data = ica.get_components()[:, idx]
    data_picks, pos, merge_grads, names, _ = _prepare_topo_plot(
        ica, ch_type, layout)
    pos, outlines = _check_outlines(pos, outlines, head_pos)
    if outlines not in (None, 'head'):
        image_mask, pos = _make_image_mask(outlines, pos, res)
    else:
        image_mask = None

    data = np.atleast_2d(data)
    data = data[:, data_picks]

    if merge_grads:
        from ..channels.layout import _merge_grad_data
        data = _merge_grad_data(data)
    axes.set_title('IC #%03d' % idx, fontsize=12)
    vmin_, vmax_ = _setup_vmin_vmax(data, vmin, vmax)
    im = plot_topomap(data.ravel(), pos, vmin=vmin_, vmax=vmax_,
                      res=res, axes=axes, cmap=cmap, outlines=outlines,
                      image_mask=image_mask, contours=contours,
                      image_interp=image_interp, show=show)[0]
    if colorbar:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid import make_axes_locatable
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%3.2f', cmap=cmap)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks((vmin_, vmax_))
        cbar.ax.set_title('AU', fontsize=10)
    _hide_frame(axes)


def plot_ica_components(ica, picks=None, ch_type=None, res=64,
                        layout=None, vmin=None, vmax=None, cmap='RdBu_r',
                        sensors=True, colorbar=False, title=None,
                        show=True, outlines='head', contours=6,
                        image_interp='bilinear', head_pos=None,
                        inst=None):
    """Project unmixing matrix on interpolated sensor topogrpahy.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    picks : int | array-like | None
        The indices of the sources to be plotted.
        If None all are plotted in batches of 20.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are
        collected in pairs and the RMS for each pair is plotted.
        If None, then channels are chosen in the order given above.
    res : int
        The resolution of the topomap image (n pixels along each side).
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True). Defaults to 'RdBu_r'.

        .. warning::  Interactive mode works smoothly only for a small amount
            of topomaps.

    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib
        plot format string (e.g., 'r+' for red plusses). If True, a circle
        will be used (via .add_artist). Defaults to True.
    colorbar : bool
        Plot a colorbar.
    title : str | None
        Title to use.
    show : bool
        Show figure if True.
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    head_pos : dict | None
        If None (default), the sensors are positioned such that they span
        the head circle. If dict, can have entries 'center' (tuple) and
        'scale' (tuple) for what the center and scale of the head should be
        relative to the electrode locations.
    inst : Raw | Epochs | None
        To be able to see component properties after clicking on component
        topomap you need to pass relevant data - instances of Raw or Epochs
        (for example the data that ICA was trained on). This takes effect
        only when running matplotlib in interactive mode.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure or list
        The figure object(s).
    """
    from ..io import BaseRaw
    from ..epochs import BaseEpochs
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid import make_axes_locatable
    from ..channels import _get_ch_type

    if picks is None:  # plot components by sets of 20
        ch_type = _get_ch_type(ica, ch_type)
        n_components = ica.mixing_matrix_.shape[1]
        p = 20
        figs = []
        for k in range(0, n_components, p):
            picks = range(k, min(k + p, n_components))
            fig = plot_ica_components(ica, picks=picks, ch_type=ch_type,
                                      res=res, layout=layout, vmax=vmax,
                                      cmap=cmap, sensors=sensors,
                                      colorbar=colorbar, title=title,
                                      show=show, outlines=outlines,
                                      contours=contours,
                                      image_interp=image_interp,
                                      head_pos=head_pos, inst=inst)
            figs.append(fig)
        return figs
    elif np.isscalar(picks):
        picks = [picks]
    ch_type = _get_ch_type(ica, ch_type)

    cmap = _setup_cmap(cmap, n_axes=len(picks))
    data = np.dot(ica.mixing_matrix_[:, picks].T,
                  ica.pca_components_[:ica.n_components_])

    if ica.info is None:
        raise RuntimeError('The ICA\'s measurement info is missing. Please '
                           'fit the ICA or add the corresponding info object.')

    data_picks, pos, merge_grads, names, _ = _prepare_topo_plot(ica, ch_type,
                                                                layout)
    pos, outlines = _check_outlines(pos, outlines, head_pos)
    if outlines not in (None, 'head'):
        image_mask, pos = _make_image_mask(outlines, pos, res)
    else:
        image_mask = None

    data = np.atleast_2d(data)
    data = data[:, data_picks]

    # prepare data for iteration
    fig, axes = _prepare_trellis(len(data), max_col=5)
    if title is None:
        title = 'ICA components'
    fig.suptitle(title)

    if merge_grads:
        from ..channels.layout import _merge_grad_data
    for ii, data_, ax in zip(picks, data, axes):
        ax.set_title('IC #%03d' % ii, fontsize=12)
        data_ = _merge_grad_data(data_) if merge_grads else data_
        vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
        im = plot_topomap(data_.flatten(), pos, vmin=vmin_, vmax=vmax_,
                          res=res, axes=ax, cmap=cmap[0], outlines=outlines,
                          image_mask=image_mask, contours=contours,
                          image_interp=image_interp, show=False)[0]
        im.axes.set_label('IC #%03d' % ii)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%3.2f', cmap=cmap)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_ticks((vmin_, vmax_))
            cbar.ax.set_title('AU', fontsize=10)
            if cmap[1]:
                ax.CB = DraggableColorbar(cbar, im)
        _hide_frame(ax)
    tight_layout(fig=fig)
    fig.subplots_adjust(top=0.95)
    fig.canvas.draw()
    if isinstance(inst, (BaseRaw, BaseEpochs)):
        def onclick(event, ica=ica, inst=inst):
            # check which component to plot
            label = event.inaxes.get_label()
            if 'IC #' in label:
                ic = int(label[4:])
                ica.plot_properties(inst, picks=ic, show=True)
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt_show(show)
    return fig


def plot_tfr_topomap(tfr, tmin=None, tmax=None, fmin=None, fmax=None,
                     ch_type=None, baseline=None, mode='mean', layout=None,
                     vmin=None, vmax=None, cmap=None, sensors=True,
                     colorbar=True, unit=None, res=64, size=2,
                     cbar_fmt='%1.1e', show_names=False, title=None,
                     axes=None, show=True, outlines='head', head_pos=None):
    """Plot topographic maps of specific time-frequency intervals of TFR data.

    Parameters
    ----------
    tfr : AvereageTFR
        The AvereageTFR object.
    tmin : None | float
        The first time instant to display. If None the first time point
        available is used.
    tmax : None | float
        The last time instant to display. If None the last time point
        available is used.
    fmin : None | float
        The first frequency to display. If None the first frequency
        available is used.
    fmax : None | float
        The last frequency to display. If None the last frequency
        available is used.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are
        collected in pairs and the RMS for each pair is plotted.
        If None, then channels are chosen in the order given above.
    baseline : tuple or list of length 2
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    mode : 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or z-score (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
        If None, baseline no correction will be performed.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout
        file is inferred from the data; if no appropriate layout file
        was found, the layout is automatically generated from the sensor
        locations.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data) or in case
        data contains only positive values 0. If callable, the output equals
        vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range. If None, the
        maximum value is used. If callable, the output equals vmax(data).
        Defaults to None.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None (default), 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True).
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib
        plot format string (e.g., 'r+' for red plusses). If True, a circle will
        be used (via .add_artist). Defaults to True.
    colorbar : bool
        Plot a colorbar.
    unit : str | None
        The unit of the channel type used for colorbar labels.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : float
        Side length per topomap in inches (only applies when plotting multiple
        topomaps at a time).
    cbar_fmt : str
        String format for colorbar values.
    show_names : bool | callable
        If True, show channel names on top of the map. If a callable is
        passed, channel names will be formatted using the callable; e.g., to
        delete the prefix 'MEG ' from all channel names, pass the function
        lambda x: x.replace('MEG ', ''). If `mask` is not None, only
        significant sensors will be shown.
    title : str | None
        Title. If None (default), no title is displayed.
    axes : instance of Axis | None
        The axes to plot to. If None the axes is defined automatically.
    show : bool
        Show figure if True.
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    head_pos : dict | None
        If None (default), the sensors are positioned such that they span
        the head circle. If dict, can have entries 'center' (tuple) and
        'scale' (tuple) for what the center and scale of the head should be
        relative to the electrode locations.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the topography.
    """
    from ..channels import _get_ch_type
    ch_type = _get_ch_type(tfr, ch_type)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    picks, pos, merge_grads, names, _ = _prepare_topo_plot(tfr, ch_type,
                                                           layout)
    if not show_names:
        names = None

    data = tfr.data
    data = rescale(data, tfr.times, baseline, mode, copy=True)

    # crop time
    itmin, itmax = None, None
    idx = np.where(_time_mask(tfr.times, tmin, tmax))[0]
    if tmin is not None:
        itmin = idx[0]
    if tmax is not None:
        itmax = idx[-1] + 1

    # crop freqs
    ifmin, ifmax = None, None
    idx = np.where(_time_mask(tfr.freqs, fmin, fmax))[0]
    if fmin is not None:
        ifmin = idx[0]
    if fmax is not None:
        ifmax = idx[-1] + 1

    data = data[picks, ifmin:ifmax, itmin:itmax]
    data = np.mean(np.mean(data, axis=2), axis=1)[:, np.newaxis]

    if merge_grads:
        from ..channels.layout import _merge_grad_data
        data = _merge_grad_data(data)

    norm = False if np.min(data) < 0 else True
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    cmap = _setup_cmap(cmap, norm=norm)

    if axes is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = axes.figure
        ax = axes

    _hide_frame(ax)

    if title is not None:
        ax.set_title(title)
    fig_wrapper = list()
    selection_callback = partial(_onselect, tfr=tfr, pos=pos, ch_type=ch_type,
                                 itmin=itmin, itmax=itmax, ifmin=ifmin,
                                 ifmax=ifmax, cmap=cmap[0], fig=fig_wrapper,
                                 layout=layout)

    im, _ = plot_topomap(data[:, 0], pos, vmin=vmin, vmax=vmax,
                         axes=ax, cmap=cmap[0], image_interp='bilinear',
                         contours=False, names=names, show_names=show_names,
                         show=False, onselect=selection_callback,
                         sensors=sensors, res=res, head_pos=head_pos,
                         outlines=outlines)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format=cbar_fmt, cmap=cmap[0])
        cbar.set_ticks((vmin, vmax))
        cbar.ax.tick_params(labelsize=12)
        unit = _handle_default('units', unit)['misc']
        cbar.ax.set_title(unit)
        if cmap[1]:
            ax.CB = DraggableColorbar(cbar, im)

    plt_show(show)
    return fig


def plot_evoked_topomap(evoked, times="auto", ch_type=None, layout=None,
                        vmin=None, vmax=None, cmap=None, sensors=True,
                        colorbar=True, scale=None, scale_time=1e3, unit=None,
                        res=64, size=1, cbar_fmt='%3.1f',
                        time_format='%01d ms', proj=False, show=True,
                        show_names=False, title=None, mask=None,
                        mask_params=None, outlines='head', contours=6,
                        image_interp='bilinear', average=None, head_pos=None,
                        axes=None):
    """Plot topographic maps of specific time points of evoked data.

    Parameters
    ----------
    evoked : Evoked
        The Evoked object.
    times : float | array of floats | "auto" | "peaks".
        The time point(s) to plot. If "auto", the number of ``axes`` determines
        the amount of time point(s). If ``axes`` is also None, at most 10
        topographies will be shown with a regular time spacing between the
        first and last time instant. If "peaks", finds time points
        automatically by checking for local maxima in global field power.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted.
        If None, then channels are chosen in the order given above.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout file
        is inferred from the data; if no appropriate layout file was found, the
        layout is automatically generated from the sensor locations.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range (zoom).
        The mouse scroll can also be used to adjust the range. Hitting space
        bar resets the range. Up and down arrows can be used to change the
        colormap. If None (default), 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True).

        .. warning::  Interactive mode works smoothly only for a small amount
            of topomaps. Interactive mode is disabled by default for more than
            2 topomaps.

    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    colorbar : bool
        Plot a colorbar.
    scale : dict | float | None
        Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
        for grad and 1e15 for mag.
    scale_time : float | None
        Scale the time labels. Defaults to 1e3 (ms).
    unit : dict | str | None
        The unit of the channel type used for colorbar label. If
        scale is None the unit is automatically determined.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : float
        Side length per topomap in inches.
    cbar_fmt : str
        String format for colorbar values.
    time_format : str
        String format for topomap values. Defaults to "%01d ms"
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be show.
    show : bool
        Show figure if True.
    show_names : bool | callable
        If True, show channel names on top of the map. If a callable is
        passed, channel names will be formatted using the callable; e.g., to
        delete the prefix 'MEG ' from all channel names, pass the function
        lambda x: x.replace('MEG ', ''). If `mask` is not None, only
        significant sensors will be shown.
    title : str | None
        Title. If None (default), no title is displayed.
    mask : ndarray of bool, shape (n_channels, n_times) | None
        The channels to be marked as significant at a given time point.
        Indices set to `True` will be considered. Defaults to None.
    mask_params : dict | None
        Additional plotting parameters for plotting significant sensors.
        Default (None) equals::

            dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                 linewidth=0, markersize=4)

    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | False | array of float | None
        The number of contour lines to draw. If 0, no contours will be drawn.
        If an array, the values represent the levels for the contours. The
        values are in uV for EEG, fT for magnetometers and fT/m for
        gradiometers. If colorbar=True, the ticks in colorbar correspond to the
        contour levels.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    average : float | None
        The time window around a given time to be used for averaging (seconds).
        For example, 0.01 would translate into window that starts 5 ms before
        and ends 5 ms after a given time point. Defaults to None, which means
        no averaging.
    head_pos : dict | None
        If None (default), the sensors are positioned such that they span
        the head circle. If dict, can have entries 'center' (tuple) and
        'scale' (tuple) for what the center and scale of the head should be
        relative to the electrode locations.
    axes : instance of Axes | list | None
        The axes to plot to. If list, the list must be a list of Axes of the
        same length as ``times`` (unless ``times`` is None). If instance of
        Axes, ``times`` must be a float or a list of one float.
        Defaults to None.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
       The figure.
    """
    from ..channels import _get_ch_type
    ch_type = _get_ch_type(evoked, ch_type)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: F401

    mask_params = _handle_default('mask_params', mask_params)
    mask_params['markersize'] *= size / 2.
    mask_params['markeredgewidth'] *= size / 2.

    picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
        evoked, ch_type, layout)

    # project before picks
    if proj is True and evoked.proj is not True:
        data = evoked.copy().apply_proj().data
    else:
        data = evoked.data

    evoked = evoked.copy().pick_channels(
        [evoked.ch_names[pick] for pick in picks])

    if axes is not None:
        if isinstance(axes, plt.Axes):
            axes = [axes]
        times = _process_times(evoked, times, n_peaks=len(axes))
    else:
        times = _process_times(evoked, times, n_peaks=None)
    space = 1 / (2. * evoked.info['sfreq'])
    if (max(times) > max(evoked.times) + space or
            min(times) < min(evoked.times) - space):
        raise ValueError('Times should be between {0:0.3f} and '
                         '{1:0.3f}.'.format(evoked.times[0], evoked.times[-1]))
    n_times = len(times)
    nax = n_times + bool(colorbar)
    width = size * nax
    height = size + max(0, 0.1 * (4 - size)) + bool(title) * 0.5
    if axes is None:
        plt.figure(figsize=(width, height))
        axes = list()
        for ax_idx in range(len(times)):
            if colorbar:  # Make room for the colorbar
                axes.append(plt.subplot(1, n_times + 1, ax_idx + 1))
            else:
                axes.append(plt.subplot(1, n_times, ax_idx + 1))
    elif colorbar:
        warn('Colorbar is drawn to the rightmost column of the figure. Be '
             'sure to provide enough space for it or turn it off with '
             'colorbar=False.')
    if len(axes) != n_times:
        raise RuntimeError('Axes and times must be equal in sizes.')

    if ch_type.startswith('planar'):
        key = 'grad'
    else:
        key = ch_type

    scale = _handle_default('scalings', scale)[key]
    unit = _handle_default('units', unit)[key]

    if not show_names:
        names = None

    w_frame = plt.rcParams['figure.subplot.wspace'] / (2 * nax)
    top_frame = max((0.05 if title is None else 0.25), .2 / size)
    fig = axes[0].get_figure()
    fig.subplots_adjust(left=w_frame, right=1 - w_frame, bottom=0,
                        top=1 - top_frame)
    # find first index that's >= (to rounding error) to each time point
    time_idx = [np.where(_time_mask(evoked.times, tmin=t,
                         tmax=None, sfreq=evoked.info['sfreq']))[0][0]
                for t in times]

    if average is None:
        data = data[np.ix_(picks, time_idx)]
    elif isinstance(average, float):
        if not average > 0:
            raise ValueError('The average parameter must be positive. You '
                             'passed a negative value')
        data_ = np.zeros((len(picks), len(time_idx)))
        ave_time = float(average) / 2.
        iter_times = evoked.times[time_idx]
        for ii, (idx, tmin_, tmax_) in enumerate(zip(time_idx,
                                                     iter_times - ave_time,
                                                     iter_times + ave_time)):
            my_range = (tmin_ < evoked.times) & (evoked.times < tmax_)
            data_[:, ii] = data[picks][:, my_range].mean(-1)
        data = data_
    else:
        raise ValueError('The average parameter must be None or a float.'
                         'Check your input.')

    data *= scale
    if merge_grads:
        from ..channels.layout import _merge_grad_data
        data = _merge_grad_data(data)

    images, contours_ = [], []

    if mask is not None:
        _picks = picks[::2 if ch_type not in ['mag', 'eeg'] else 1]
        mask_ = mask[np.ix_(_picks, time_idx)]

    pos, outlines = _check_outlines(pos, outlines, head_pos)
    if outlines is not None:
        image_mask, pos = _make_image_mask(outlines, pos, res)
    else:
        image_mask = None

    vlims = [_setup_vmin_vmax(data[:, i], vmin, vmax, norm=merge_grads)
             for i in range(len(times))]
    vmin = np.min(vlims)
    vmax = np.max(vlims)
    cmap = _setup_cmap(cmap, n_axes=len(times), norm=vmin >= 0)

    if not isinstance(contours, (list, np.ndarray)):
        _, contours = _set_contour_locator(vmin, vmax, contours)

    for idx, time in enumerate(times):
        tp, cn = plot_topomap(data[:, idx], pos, vmin=vmin, vmax=vmax,
                              sensors=sensors, res=res, names=names,
                              show_names=show_names, cmap=cmap[0],
                              mask=mask_[:, idx] if mask is not None else None,
                              mask_params=mask_params, axes=axes[idx],
                              outlines=outlines, image_mask=image_mask,
                              contours=contours, image_interp=image_interp,
                              show=False)

        images.append(tp)
        if cn is not None:
            contours_.append(cn)
        if time_format is not None:
            axes[idx].set_title(time_format % (time * scale_time))

    if title is not None:
        plt.suptitle(title, verticalalignment='top', size='x-large')

    if colorbar:
        # works both when fig axes pre-defined and when not
        n_fig_axes = max(nax, len(fig.get_axes()))
        cax = plt.subplot(1, n_fig_axes + 1, n_fig_axes + 1)
        # resize the colorbar (by default the color fills the whole axes)
        cpos = cax.get_position()
        if size <= 1:
            cpos.x0 = 1 - (.7 + .1 / size) / n_fig_axes
        cpos.x1 = cpos.x0 + .1 / n_fig_axes
        cpos.y0 = .2
        cpos.y1 = .7
        cax.set_position(cpos)
        if unit is not None:
            cax.set_title(unit)
        cbar = fig.colorbar(images[-1], ax=cax, cax=cax, format=cbar_fmt)
        cbar.set_ticks(cn.levels)
        cbar.ax.tick_params(labelsize=7)

        if cmap[1]:
            for im in images:
                im.axes.CB = DraggableColorbar(cbar, im)

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      picks=picks, images=images, contours=contours_,
                      time_idx=time_idx, scale=scale, merge_grads=merge_grads,
                      res=res, pos=pos, image_mask=image_mask,
                      plot_update_proj_callback=_plot_update_evoked_topomap)
        _draw_proj_checkbox(None, params)

    plt_show(show)
    return fig


def _plot_topomap_multi_cbar(data, pos, ax, title=None, unit=None, vmin=None,
                             vmax=None, cmap=None, outlines='head',
                             colorbar=False, cbar_fmt='%3.3f'):
    """Plot topomap multi cbar."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _hide_frame(ax)
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    cmap = _setup_cmap(cmap)
    if title is not None:
        ax.set_title(title, fontsize=10)
    im, _ = plot_topomap(data, pos, vmin=vmin, vmax=vmax, axes=ax,
                         cmap=cmap[0], image_interp='bilinear', contours=False,
                         outlines=outlines, show=False)

    if colorbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.25)
        cbar = plt.colorbar(im, cax=cax, format=cbar_fmt)
        cbar.set_ticks((vmin, vmax))
        if unit is not None:
            cbar.ax.set_title(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        if cmap[1]:
            ax.CB = DraggableColorbar(cbar, im)


@verbose
def plot_epochs_psd_topomap(epochs, bands=None, vmin=None, vmax=None,
                            tmin=None, tmax=None, proj=False,
                            bandwidth=None, adaptive=False, low_bias=True,
                            normalization='length', ch_type=None, layout=None,
                            cmap='RdBu_r', agg_fun=None, dB=False, n_jobs=1,
                            normalize=False, cbar_fmt='%0.3f',
                            outlines='head', axes=None, show=True,
                            verbose=None):
    """Plot the topomap of the power spectral density across epochs.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs object
    bands : list of tuple | None
        The lower and upper frequency and the name for that band. If None,
        (default) expands to:

        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]

    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None np.min(data) is used. If callable, the output equals
        vmin(data).
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    tmin : float | None
        Start time to consider.
    tmax : float | None
        End time to consider.
    proj : bool
        Apply projection.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz. The default
        value is a window half-bandwidth of 4 Hz.
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
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted. If None, then first
        available channel type from order given above is used. Defaults to
        None.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout
        file is inferred from the data; if no appropriate layout file was
        found, the layout is automatically generated from the sensor
        locations.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None (default), 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True).
    agg_fun : callable
        The function used to aggregate over frequencies.
        Defaults to np.sum. if normalize is True, else np.mean.
    dB : bool
        If True, transform data to decibels (with ``10 * np.log10(data)``)
        following the application of `agg_fun`. Only valid if normalize is
        False.
    n_jobs : int
        Number of jobs to run in parallel.
    normalize : bool
        If True, each band will be divided by the total power. Defaults to
        False.
    cbar_fmt : str
        The colorbar format. Defaults to '%0.3f'.
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    axes : list of axes | None
        List of axes to plot consecutive topographies to. If None the axes
        will be created automatically. Defaults to None.
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
    from ..channels import _get_ch_type
    ch_type = _get_ch_type(epochs, ch_type)

    picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
        epochs, ch_type, layout)

    psds, freqs = psd_multitaper(epochs, tmin=tmin, tmax=tmax,
                                 bandwidth=bandwidth, adaptive=adaptive,
                                 low_bias=low_bias,
                                 normalization=normalization, picks=picks,
                                 proj=proj, n_jobs=n_jobs)
    psds = np.mean(psds, axis=0)

    if merge_grads:
        from ..channels.layout import _merge_grad_data
        psds = _merge_grad_data(psds)

    return plot_psds_topomap(
        psds=psds, freqs=freqs, pos=pos, agg_fun=agg_fun, vmin=vmin,
        vmax=vmax, bands=bands, cmap=cmap, dB=dB, normalize=normalize,
        cbar_fmt=cbar_fmt, outlines=outlines, axes=axes, show=show)


def plot_psds_topomap(
        psds, freqs, pos, agg_fun=None, vmin=None, vmax=None, bands=None,
        cmap=None, dB=True, normalize=False, cbar_fmt='%0.3f', outlines='head',
        axes=None, show=True):
    """Plot spatial maps of PSDs.

    Parameters
    ----------
    psds : np.ndarray of float, shape (n_channels, n_freqs)
        Power spectral densities
    freqs : np.ndarray of float, shape (n_freqs)
        Frequencies used to compute psds.
    pos : numpy.ndarray of float, shape (n_sensors, 2)
        The positions of the sensors.
    agg_fun : callable
        The function used to aggregate over frequencies.
        Defaults to np.sum. if normalize is True, else np.mean.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None np.min(data) is used. If callable, the output equals
        vmin(data).
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    bands : list of tuple | None
        The lower and upper frequency and the name for that band. If None,
        (default) expands to:

            bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                     (12, 30, 'Beta'), (30, 45, 'Gamma')]

    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None (default), 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True).
    dB : bool
        If True, transform data to decibels (with ``10 * np.log10(data)``)
        following the application of `agg_fun`. Only valid if normalize is
        False.
    normalize : bool
        If True, each band will be divided by the total power. Defaults to
        False.
    cbar_fmt : str
        The colorbar format. Defaults to '%0.3f'.
    outlines : 'head' | 'skirt' | dict | None
        The outlines to be drawn. If 'head', the default head scheme will be
        drawn. If 'skirt' the head scheme will be drawn, but sensors are
        allowed to be plotted outside of the head circle. If dict, each key
        refers to a tuple of x and y positions, the values in 'mask_pos' will
        serve as image mask, and the 'autoshrink' (bool) field will trigger
        automated shrinking of the positions due to points outside the outline.
        Alternatively, a matplotlib patch object can be passed for advanced
        masking options, either directly or as a function that returns patches
        (required for multi-axis plots). If None, nothing will be drawn.
        Defaults to 'head'.
    axes : list of axes | None
        List of axes to plot consecutive topographies to. If None the axes
        will be created automatically. Defaults to None.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    import matplotlib.pyplot as plt

    if bands is None:
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]

    if agg_fun is None:
        agg_fun = np.sum if normalize is True else np.mean

    if normalize is True:
        psds /= psds.sum(axis=-1)[..., None]
        assert np.allclose(psds.sum(axis=-1), 1.)

    n_axes = len(bands)
    if axes is not None:
        _validate_if_list_of_axes(axes, n_axes)
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, n_axes, figsize=(2 * n_axes, 1.5))
        if n_axes == 1:
            axes = [axes]

    for ax, (fmin, fmax, title) in zip(axes, bands):
        freq_mask = (fmin < freqs) & (freqs < fmax)
        if freq_mask.sum() == 0:
            raise RuntimeError('No frequencies in band "%s" (%s, %s)'
                               % (title, fmin, fmax))
        data = agg_fun(psds[:, freq_mask], axis=1)
        if dB is True and normalize is False:
            data = 10 * np.log10(data)
            unit = 'dB'
        else:
            unit = 'power'

        _plot_topomap_multi_cbar(data, pos, ax, title=title, vmin=vmin,
                                 vmax=vmax, cmap=cmap, outlines=outlines,
                                 colorbar=True, unit=unit, cbar_fmt=cbar_fmt)
    tight_layout(fig=fig)
    fig.canvas.draw()
    plt_show(show)
    return fig


def plot_layout(layout, show=True):
    """Plot the sensor positions.

    Parameters
    ----------
    layout : None | Layout
        Layout instance specifying sensor positions.
    show : bool
        Show figure if True. Defaults to True.

    Returns
    -------
    fig : instance of matplotlib figure
        Figure containing the sensor topography.

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None,
                        hspace=None)
    ax.set_xticks([])
    ax.set_yticks([])
    pos = [(p[0] + p[2] / 2., p[1] + p[3] / 2.) for p in layout.pos]
    pos, outlines = _check_outlines(pos, 'head')
    _draw_outlines(ax, outlines)
    for ii, (this_pos, ch_id) in enumerate(zip(pos, layout.names)):
        ax.annotate(ch_id, xy=this_pos[:2], horizontalalignment='center',
                    verticalalignment='center', size='x-small')
    plt_show(show)
    return fig


def _onselect(eclick, erelease, tfr, pos, ch_type, itmin, itmax, ifmin, ifmax,
              cmap, fig, layout=None):
    """Callback called from topomap for drawing average tfr over channels."""
    import matplotlib.pyplot as plt
    pos, _ = _check_outlines(pos, outlines='head', head_pos=None)
    ax = eclick.inaxes
    xmin = min(eclick.xdata, erelease.xdata)
    xmax = max(eclick.xdata, erelease.xdata)
    ymin = min(eclick.ydata, erelease.ydata)
    ymax = max(eclick.ydata, erelease.ydata)
    indices = [i for i in range(len(pos)) if pos[i][0] < xmax and
               pos[i][0] > xmin and pos[i][1] < ymax and pos[i][1] > ymin]
    for idx, circle in enumerate(ax.artists):
        if idx in indices:
            circle.set_color('r')
        else:
            circle.set_color('black')
    plt.gcf().canvas.draw()
    if not indices:
        return
    data = tfr.data
    if ch_type == 'mag':
        picks = pick_types(tfr.info, meg=ch_type, ref_meg=False)
        data = np.mean(data[indices, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[picks[x]] for x in indices]
    elif ch_type == 'grad':
        from ..channels.layout import _pair_grad_sensors
        grads = _pair_grad_sensors(tfr.info, layout=layout,
                                   topomap_coords=False)
        idxs = list()
        for idx in indices:
            idxs.append(grads[idx * 2])
            idxs.append(grads[idx * 2 + 1])  # pair of grads
        data = np.mean(data[idxs, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[x] for x in idxs]
    elif ch_type == 'eeg':
        picks = pick_types(tfr.info, meg=False, eeg=True, ref_meg=False)
        data = np.mean(data[indices, ifmin:ifmax, itmin:itmax], axis=0)
        chs = [tfr.ch_names[picks[x]] for x in indices]
    logger.info('Averaging TFR over channels ' + str(chs))
    if len(fig) == 0:
        fig.append(figure_nobar())
    if not plt.fignum_exists(fig[0].number):
        fig[0] = figure_nobar()
    ax = fig[0].add_subplot(111)
    itmax = len(tfr.times) - 1 if itmax is None else min(itmax,
                                                         len(tfr.times) - 1)
    ifmax = len(tfr.freqs) - 1 if ifmax is None else min(ifmax,
                                                         len(tfr.freqs) - 1)
    if itmin is None:
        itmin = 0
    if ifmin is None:
        ifmin = 0
    extent = (tfr.times[itmin] * 1e3, tfr.times[itmax] * 1e3, tfr.freqs[ifmin],
              tfr.freqs[ifmax])

    title = 'Average over %d %s channels.' % (len(chs), ch_type)
    ax.set_title(title)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    img = ax.imshow(data, extent=extent, aspect="auto", origin="lower",
                    cmap=cmap)
    if len(fig[0].get_axes()) < 2:
        fig[0].get_axes()[1].cbar = fig[0].colorbar(mappable=img)
    else:
        fig[0].get_axes()[1].cbar.on_mappable_changed(mappable=img)
    fig[0].canvas.draw()
    plt.figure(fig[0].number)
    plt_show(True)


def _prepare_topomap(pos, ax):
    """Helper for preparing the topomap."""
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    _hide_frame(ax)
    if any([not pos_y.any(), not pos_x.any()]):
        raise RuntimeError('No position information found, cannot compute '
                           'geometries for topomap.')
    return pos_x, pos_y


def _hide_frame(ax):
    """Helper to hide axis frame for topomaps."""
    ax.get_yticks()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)


def _init_anim(ax, ax_line, ax_cbar, params, merge_grads):
    """Initialize animated topomap."""
    from matplotlib import pyplot as plt, patches
    logger.info('Initializing animation...')
    data = params['data']
    items = list()
    if params['butterfly']:
        all_times = params['all_times']
        for idx in range(len(data)):
            ax_line.plot(all_times, data[idx], color='k')
        vmin, vmax = _setup_vmin_vmax(data, None, None)
        ax_line.set_yticks(np.around(np.linspace(vmin, vmax, 5), -1))
        params['line'], = ax_line.plot([all_times[0], all_times[0]],
                                       ax_line.get_ylim(), color='r')
        items.append(params['line'])
    if merge_grads:
        from mne.channels.layout import _merge_grad_data
        data = _merge_grad_data(data)
    norm = True if np.min(data) > 0 else False
    cmap = 'Reds' if norm else 'RdBu_r'

    vmin, vmax = _setup_vmin_vmax(data, None, None, norm)

    pos, outlines = _check_outlines(params['pos'], 'head', None)
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]

    _hide_frame(ax)
    xlim = np.inf, -np.inf,
    ylim = np.inf, -np.inf,
    mask_ = np.c_[outlines['mask_pos']]
    xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                  np.max(np.r_[xlim[1], mask_[:, 0]]))
    ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                  np.max(np.r_[ylim[1], mask_[:, 1]]))

    res = 64
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    params['Zis'] = list()

    for frame in params['frames']:
        Zi = _griddata(pos_x, pos_y, data[:, frame], Xi, Yi)
        params['Zis'].append(Zi)
    Zi = params['Zis'][0]
    zi_min = np.min(params['Zis'])
    zi_max = np.max(params['Zis'])
    cont_lims = np.linspace(zi_min, zi_max, 7, endpoint=False)[1:]
    _, pos = _make_image_mask(outlines, pos, res)
    params.update({'vmin': vmin, 'vmax': vmax, 'Xi': Xi, 'Yi': Yi, 'Zi': Zi,
                   'extent': (xmin, xmax, ymin, ymax), 'cmap': cmap,
                   'cont_lims': cont_lims})
    # plot map and countour
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax),
                   interpolation='bilinear')
    plt.colorbar(im, cax=ax_cbar, cmap=cmap)
    cont = ax.contour(Xi, Yi, Zi, levels=cont_lims, colors='k', linewidths=1)

    patch_ = patches.Ellipse((0, 0),
                             2 * outlines['clip_radius'][0],
                             2 * outlines['clip_radius'][1],
                             clip_on=True,
                             transform=ax.transData)
    im.set_clip_path(patch_)
    text = ax.text(0.55, 0.95, '', transform=ax.transAxes, va='center',
                   ha='right')
    params['text'] = text
    items.append(im)
    items.append(text)
    for col in cont.collections:
        col.set_clip_path(patch_)

    outlines_ = _draw_outlines(ax, outlines)

    params.update({'patch': patch_, 'outlines': outlines_})
    return tuple(items) + tuple(cont.collections)


def _animate(frame, ax, ax_line, params):
    """Update animated topomap."""
    if params['pause']:
        frame = params['frame']
    time_idx = params['frames'][frame]

    title = '%6.0f ms' % (params['times'][frame] * 1e3)
    if params['blit']:
        text = params['text']
    else:
        ax.cla()  # Clear old contours.
        text = ax.text(0.45, 1.15, '', transform=ax.transAxes)
        for k, (x, y) in params['outlines'].items():
            if 'mask' in k:
                continue
            ax.plot(x, y, color='k', linewidth=1, clip_on=False)

    _hide_frame(ax)
    text.set_text(title)

    vmin = params['vmin']
    vmax = params['vmax']
    Xi = params['Xi']
    Yi = params['Yi']
    Zi = params['Zis'][frame]
    extent = params['extent']
    cmap = params['cmap']
    patch = params['patch']

    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=extent, interpolation='bilinear')
    cont_lims = params['cont_lims']
    cont = ax.contour(Xi, Yi, Zi, levels=cont_lims, colors='k', linewidths=1)

    im.set_clip_path(patch)
    items = [im, text]
    for col in cont.collections:
        col.set_clip_path(patch)

    if params['butterfly']:
        all_times = params['all_times']
        line = params['line']
        line.remove()
        params['line'] = ax_line.plot([all_times[time_idx],
                                       all_times[time_idx]],
                                      ax_line.get_ylim(), color='r')[0]
        items.append(params['line'])
    params['frame'] = frame
    return tuple(items) + tuple(cont.collections)


def _pause_anim(event, params):
    """Pause or continue the animation on mouse click."""
    params['pause'] = not params['pause']


def _key_press(event, params):
    """Function for handling key presses for the animation."""
    if event.key == 'left':
        params['pause'] = True
        params['frame'] = max(params['frame'] - 1, 0)
    elif event.key == 'right':
        params['pause'] = True
        params['frame'] = min(params['frame'] + 1, len(params['frames']) - 1)


def _topomap_animation(evoked, ch_type='mag', times=None, frame_rate=None,
                       butterfly=False, blit=True, show=True):
    """Make animation of evoked data as topomap timeseries.

    Animation can be paused/resumed with left mouse button.
    Left and right arrow keys can be used to move backward or forward in
    time.

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
        the animation it is better to disable blit. For MacOSX blit is always
        disabled. Defaults to True.
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
    from matplotlib import pyplot as plt, animation
    if ch_type is None:
        ch_type = _picks_by_type(evoked.info)[0][0]
    if ch_type not in ('mag', 'grad', 'eeg'):
        raise ValueError("Channel type not supported. Supported channel "
                         "types include 'mag', 'grad' and 'eeg'.")
    if times is None:
        times = np.linspace(evoked.times[0], evoked.times[-1], 10)
    times = np.array(times)

    if times.ndim != 1:
        raise ValueError('times must be 1D, got %d dimensions' % times.ndim)
    if max(times) > evoked.times[-1] or min(times) < evoked.times[0]:
        raise ValueError('All times must be inside the evoked time series.')
    frames = [np.abs(evoked.times - time).argmin() for time in times]

    blit = False if plt.get_backend() == 'MacOSX' else blit
    picks, pos, merge_grads, _, ch_type = _prepare_topo_plot(evoked,
                                                             ch_type=ch_type,
                                                             layout=None)
    data = evoked.data[picks, :]
    data *= _handle_default('scalings')[ch_type]

    fig = plt.figure()
    offset = 0. if blit else 0.4  # XXX: blit changes the sizes for some reason
    ax = plt.axes([0. + offset / 2., 0. + offset / 2., 1. - offset,
                   1. - offset], xlim=(-1, 1), ylim=(-1, 1))
    if butterfly:
        ax_line = plt.axes([0.2, 0.05, 0.6, 0.1], xlim=(evoked.times[0],
                                                        evoked.times[-1]))
    else:
        ax_line = None
    if isinstance(frames, int):
        frames = np.linspace(0, len(evoked.times) - 1, frames).astype(int)
    ax_cbar = plt.axes([0.85, 0.1, 0.05, 0.8])
    ax_cbar.set_title(_handle_default('units')[ch_type], fontsize=10)

    params = {'data': data, 'pos': pos, 'all_times': evoked.times, 'frame': 0,
              'frames': frames, 'butterfly': butterfly, 'blit': blit,
              'pause': False, 'times': times}
    init_func = partial(_init_anim, ax=ax, ax_cbar=ax_cbar, ax_line=ax_line,
                        params=params, merge_grads=merge_grads)
    animate_func = partial(_animate, ax=ax, ax_line=ax_line, params=params)
    pause_func = partial(_pause_anim, params=params)
    fig.canvas.mpl_connect('button_press_event', pause_func)
    key_press_func = partial(_key_press, params=params)
    fig.canvas.mpl_connect('key_press_event', key_press_func)
    if frame_rate is None:
        frame_rate = evoked.info['sfreq'] / 10.
    interval = 1000 / frame_rate  # interval is in ms
    anim = animation.FuncAnimation(fig, animate_func, init_func=init_func,
                                   frames=len(frames), interval=interval,
                                   blit=blit)
    fig.mne_animation = anim  # to make sure anim is not garbage collected
    plt_show(show, block=False)
    if 'line' in params:
        # Finally remove the vertical line so it does not appear in saved fig.
        params['line'].remove()

    return fig, anim


def _set_contour_locator(vmin, vmax, contours):
    """Function for setting correct contour levels."""
    locator = None
    if isinstance(contours, int):
        from matplotlib import ticker
        # nbins = ticks - 1, since 2 of the ticks are vmin and vmax, the
        # correct number of bins is equal to contours + 1.
        locator = ticker.MaxNLocator(nbins=contours + 1)
        contours = locator.tick_values(vmin, vmax)
    return locator, contours
