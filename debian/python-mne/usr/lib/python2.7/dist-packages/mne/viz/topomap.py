"""Functions to plot M/EEG data e.g. topographies
"""
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
from ..io.pick import pick_types
from ..utils import _clean_names, _time_mask, verbose, logger
from .utils import (tight_layout, _setup_vmin_vmax, _prepare_trellis,
                    _check_delayed_ssp, _draw_proj_checkbox, figure_nobar)
from ..time_frequency import compute_epochs_psd
from ..defaults import _handle_default
from ..channels.layout import _find_topomap_coords
from ..fixes import _get_argrelmax


def _prepare_topo_plot(inst, ch_type, layout):
    """"Aux Function"""
    info = copy.deepcopy(inst.info)

    if layout is None and ch_type is not 'eeg':
        from ..channels import find_layout
        layout = find_layout(info)
    elif layout == 'auto':
        layout = None

    info['ch_names'] = _clean_names(info['ch_names'])
    for ii, this_ch in enumerate(info['chs']):
        this_ch['ch_name'] = info['ch_names'][ii]

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
                    logger.warning('Failed to locate %s channel positions from'
                                   ' layout. Inferring channel positions from '
                                   'data.' % ch_type)
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
    """ Helper to update topomaps """
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


def plot_projs_topomap(projs, layout=None, cmap='RdBu_r', sensors=True,
                       colorbar=False, res=64, size=1, show=True,
                       outlines='head', contours=6, image_interp='bilinear',
                       axes=None):
    """Plot topographic maps of SSP projections

    Parameters
    ----------
    projs : list of Projection
        The projections
    layout : None | Layout | list of Layout
        Layout instance specifying sensor positions (does not need to be
        specified for Neuromag data). Or a list of Layout if projections
        are from different sensor types.
    cmap : matplotlib colormap
        Colormap.
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

    if layout is None:
        from ..channels import read_layout
        layout = read_layout('Vectorview-all')

    if not isinstance(layout, list):
        layout = [layout]

    n_projs = len(projs)
    nrows = math.floor(math.sqrt(n_projs))
    ncols = math.ceil(n_projs / nrows)

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
        ch_names = _clean_names(proj['data']['col_names'])
        data = proj['data']['data'].ravel()

        idx = []
        for l in layout:
            is_vv = l.kind.startswith('Vectorview')
            if is_vv:
                from ..channels.layout import _pair_grad_sensors_from_ch_names
                grad_pairs = _pair_grad_sensors_from_ch_names(ch_names)
                if grad_pairs:
                    ch_names = [ch_names[i] for i in grad_pairs]

            idx = [l.names.index(c) for c in ch_names if c in l.names]
            if len(idx) == 0:
                continue

            pos = l.pos[idx]
            if is_vv and grad_pairs:
                from ..channels.layout import _merge_grad_data
                shape = (len(idx) / 2, 2, -1)
                pos = pos.reshape(shape).mean(axis=1)
                data = _merge_grad_data(data[grad_pairs]).ravel()

            break

        if len(idx):
            plot_topomap(data, pos[:, :2], vmax=None, cmap=cmap,
                         sensors=sensors, res=res, axis=axes[proj_idx],
                         outlines=outlines, contours=contours,
                         image_interp=image_interp, show=False)
            if colorbar:
                plt.colorbar()
        else:
            raise RuntimeError('Cannot find a proper layout for projection %s'
                               % proj['desc'])
    tight_layout(fig=axes[0].get_figure())
    if show and plt.get_backend() != 'agg':
        plt.show()

    return axes[0].get_figure()


def _check_outlines(pos, outlines, head_pos=None):
    """Check or create outlines for topoplot
    """
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


def _griddata(x, y, v, xi, yi):
    """Aux function"""
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
    """Aux function"""
    from matplotlib.patches import Circle
    if sensors is True:
        for x, y in zip(pos_x, pos_y):
            ax.add_artist(Circle(xy=(x, y), radius=0.003, color='k'))
    else:
        ax.plot(pos_x, pos_y, sensors)


def plot_topomap(data, pos, vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                 res=64, axis=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', image_mask=None,
                 contours=6, image_interp='bilinear', show=True,
                 head_pos=None, onselect=None):
    """Plot a topographic map as image

    Parameters
    ----------
    data : array, length = n_points
        The data values to plot.
    pos : array, shape = (n_points, 2)
        For each data point, the x and y coordinates.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True, a circle will be
        used (via .add_artist). Defaults to True.
    res : int
        The resolution of the topomap image (n pixels along each side).
    axis : instance of Axis | None
        The axis to plot to. If None, the current axis will be used.
    names : list | None
        List of channel names. If None, channel names are not plotted.
    show_names : bool | callable
        If True, show channel names on top of the map. If a callable is
        passed, channel names will be formatted using the callable; e.g., to
        delete the prefix 'MEG ' from all channel names, pass the function
        lambda x: x.replace('MEG ', ''). If `mask` is not None, only
        significant sensors will be shown.
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
    image_mask : ndarray of bool, shape (res, res) | None
        The image mask to cover the interpolated surface. If None, it will be
        computed from the outline.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
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

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

    pos, outlines = _check_outlines(pos, outlines, head_pos)
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]

    ax = axis if axis else plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if any([not pos_y.any(), not pos_x.any()]):
        raise RuntimeError('No position information found, cannot compute '
                           'geometries for topomap.')
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
    if contours in (False, None):
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
        # ax.set_clip_path(patch_)
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
        outlines_ = dict([(k, v) for k, v in outlines.items() if k not in
                          ['patch', 'autoshrink']])
        for k, (x, y) in outlines_.items():
            if 'mask' in k:
                continue
            ax.plot(x, y, color='k', linewidth=linewidth, clip_on=False)

    if show_names:
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
    if show:
        plt.show()
    return im, cont


def _make_image_mask(outlines, pos, res):
    """Aux function
    """

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
    """Aux function"""
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


def plot_ica_components(ica, picks=None, ch_type=None, res=64,
                        layout=None, vmin=None, vmax=None, cmap='RdBu_r',
                        sensors=True, colorbar=False, title=None,
                        show=True, outlines='head', contours=6,
                        image_interp='bilinear', head_pos=None):
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
    cmap : matplotlib colormap
        Colormap.
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

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure or list
        The figure object(s).
    """
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
            fig = plot_ica_components(ica, picks=picks,
                                      ch_type=ch_type, res=res, layout=layout,
                                      vmax=vmax, cmap=cmap, sensors=sensors,
                                      colorbar=colorbar, title=title,
                                      show=show, outlines=outlines,
                                      contours=contours,
                                      image_interp=image_interp)
            figs.append(fig)
        return figs
    elif np.isscalar(picks):
        picks = [picks]
    ch_type = _get_ch_type(ica, ch_type)

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
                          res=res, axis=ax, cmap=cmap, outlines=outlines,
                          image_mask=image_mask, contours=contours,
                          image_interp=image_interp, show=False)[0]
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%3.2f', cmap=cmap)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_ticks((vmin_, vmax_))
            cbar.ax.set_title('AU', fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)
    tight_layout(fig=fig)
    fig.subplots_adjust(top=0.95)
    fig.canvas.draw()

    if show is True:
        plt.show()
    return fig


def plot_tfr_topomap(tfr, tmin=None, tmax=None, fmin=None, fmax=None,
                     ch_type=None, baseline=None, mode='mean', layout=None,
                     vmin=None, vmax=None, cmap=None, sensors=True,
                     colorbar=True, unit=None, res=64, size=2,
                     cbar_fmt='%1.1e', show_names=False, title=None,
                     axes=None, show=True, outlines='head', head_pos=None):
    """Plot topographic maps of specific time-frequency intervals of TFR data

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
    cmap : matplotlib colormap | None
        Colormap. If None and the plotted data is all positive, defaults to
        'Reds'. If None and data contains also negative values, defaults to
        'RdBu_r'. Defaults to None.
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
        Side length per topomap in inches.
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

    if mode is not None and baseline is not None:
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
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    if axes is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = axes.figure
        ax = axes

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)

    if title is not None:
        ax.set_title(title)
    fig_wrapper = list()
    selection_callback = partial(_onselect, tfr=tfr, pos=pos, ch_type=ch_type,
                                 itmin=itmin, itmax=itmax, ifmin=ifmin,
                                 ifmax=ifmax, cmap=cmap, fig=fig_wrapper,
                                 layout=layout)

    im, _ = plot_topomap(data[:, 0], pos, vmin=vmin, vmax=vmax,
                         axis=ax, cmap=cmap, image_interp='bilinear',
                         contours=False, names=names, show_names=show_names,
                         show=False, onselect=selection_callback)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format=cbar_fmt, cmap=cmap)
        cbar.set_ticks((vmin, vmax))
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title('AU')

    if show:
        plt.show()

    return fig


def plot_evoked_topomap(evoked, times="auto", ch_type=None, layout=None,
                        vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                        colorbar=True, scale=None, scale_time=1e3, unit=None,
                        res=64, size=1, cbar_fmt='%3.1f',
                        time_format='%01d ms', proj=False, show=True,
                        show_names=False, title=None, mask=None,
                        mask_params=None, outlines='head', contours=6,
                        image_interp='bilinear', average=None, head_pos=None,
                        axes=None):
    """Plot topographic maps of specific time points of evoked data

    Parameters
    ----------
    evoked : Evoked
        The Evoked object.
    times : float | array of floats | "auto" | "peaks".
        The time point(s) to plot. If "auto", the number of ``axes`` determines
        the amount of time point(s). If ``axes`` is also None, 10 topographies
        will be shown with a regular time spacing between the first and last
        time instant. If "peaks", finds time points automatically by checking
        for local maxima in global field power.
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
    cmap : matplotlib colormap
        Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
        'Reds'.
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
        Indicies set to `True` will be considered. Defaults to None.
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
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
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
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa

    mask_params = _handle_default('mask_params', mask_params)
    mask_params['markersize'] *= size / 2.
    mask_params['markeredgewidth'] *= size / 2.

    if isinstance(axes, plt.Axes):
        axes = [axes]

    if times == "peaks":
        npeaks = 10 if axes is None else len(axes)
        times = _find_peaks(evoked, npeaks)
    elif times == "auto":
        if axes is None:
            times = np.linspace(evoked.times[0], evoked.times[-1], 10)
        else:
            times = np.linspace(evoked.times[0], evoked.times[-1], len(axes))
    elif np.isscalar(times):
        times = [times]

    times = np.array(times)

    if times.ndim != 1:
        raise ValueError('times must be 1D, got %d dimensions' % times.ndim)
    if len(times) > 20:
        raise RuntimeError('Too many plots requested. Please pass fewer '
                           'than 20 time instants.')

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
        logger.warning('Colorbar is drawn to the rightmost column of the '
                       'figure.\nBe sure to provide enough space for it '
                       'or turn it off with colorbar=False.')
    if len(axes) != n_times:
        raise RuntimeError('Axes and times must be equal in sizes.')
    tmin, tmax = evoked.times[[0, -1]]
    _time_comp = _time_mask(times=times, tmin=tmin,  tmax=tmax)
    if not np.all(_time_comp):
        raise ValueError('Times should be between {0:0.3f} and {1:0.3f}. (Got '
                         '{2}).'.format(tmin, tmax,
                                        ['%03.f' % t
                                         for t in times[_time_comp]]))

    picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
        evoked, ch_type, layout)

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
    time_idx = [np.where(evoked.times >= t)[0][0] for t in times]

    if proj is True and evoked.proj is not True:
        data = evoked.copy().apply_proj().data
    else:
        data = evoked.data
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

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

    images, contours_ = [], []

    if mask is not None:
        _picks = picks[::2 if ch_type not in ['mag', 'eeg'] else 1]
        mask_ = mask[np.ix_(_picks, time_idx)]

    pos, outlines = _check_outlines(pos, outlines, head_pos)
    if outlines is not None:
        image_mask, pos = _make_image_mask(outlines, pos, res)
    else:
        image_mask = None

    for idx, time in enumerate(times):
        tp, cn = plot_topomap(data[:, idx], pos, vmin=vmin, vmax=vmax,
                              sensors=sensors, res=res, names=names,
                              show_names=show_names, cmap=cmap,
                              mask=mask_[:, idx] if mask is not None else None,
                              mask_params=mask_params, axis=axes[idx],
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
        tight_layout(pad=size, fig=fig)

    if colorbar:
        cax = plt.subplot(1, n_times + 1, n_times + 1)
        # resize the colorbar (by default the color fills the whole axes)
        cpos = cax.get_position()
        if size <= 1:
            cpos.x0 = 1 - (.7 + .1 / size) / nax
        cpos.x1 = cpos.x0 + .1 / nax
        cpos.y0 = .2
        cpos.y1 = .7
        cax.set_position(cpos)
        if unit is not None:
            cax.set_title(unit)
        cbar = fig.colorbar(images[-1], ax=cax, cax=cax, format=cbar_fmt)
        cbar.set_ticks([vmin, 0, vmax])

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      picks=picks, images=images, contours=contours_,
                      time_idx=time_idx, scale=scale, merge_grads=merge_grads,
                      res=res, pos=pos, image_mask=image_mask,
                      plot_update_proj_callback=_plot_update_evoked_topomap)
        _draw_proj_checkbox(None, params)

    if show:
        plt.show()

    return fig


def _plot_topomap_multi_cbar(data, pos, ax, title=None, unit=None,
                             vmin=None, vmax=None, cmap='RdBu_r',
                             colorbar=False, cbar_fmt='%3.3f'):
    """Aux Function"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    if title is not None:
        ax.set_title(title, fontsize=10)
    im, _ = plot_topomap(data, pos, vmin=vmin, vmax=vmax, axis=ax,
                         cmap=cmap, image_interp='bilinear', contours=False,
                         show=False)

    if colorbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.25)
        cbar = plt.colorbar(im, cax=cax, format=cbar_fmt)
        cbar.set_ticks((vmin, vmax))
        if unit is not None:
            cbar.ax.set_title(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)


@verbose
def plot_epochs_psd_topomap(epochs, bands=None, vmin=None, vmax=None,
                            tmin=None, tmax=None,
                            proj=False, n_fft=256, ch_type=None,
                            n_overlap=0, layout=None,
                            cmap='RdBu_r', agg_fun=None, dB=False, n_jobs=1,
                            normalize=False, cbar_fmt='%0.3f',
                            outlines='head', show=True, verbose=None):
    """Plot the topomap of the power spectral density across epochs

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
    n_fft : int
        Number of points to use in Welch FFT calculations.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted.
        If None, then channels are chosen in the order given above.
    n_overlap : int
        The number of points of overlap between blocks.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout
        file is inferred from the data; if no appropriate layout file was
        found, the layout is automatically generated from the sensor
        locations.
    cmap : matplotlib colormap
        Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
        'Reds'.
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
        If True, each band will be devided by the total power. Defaults to
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
    show : bool
        Show figure if True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    from ..channels import _get_ch_type
    ch_type = _get_ch_type(epochs, ch_type)

    picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
        epochs, ch_type, layout)

    psds, freqs = compute_epochs_psd(epochs, picks=picks, n_fft=n_fft,
                                     tmin=tmin, tmax=tmax,
                                     n_overlap=n_overlap, proj=proj,
                                     n_jobs=n_jobs)
    psds = np.mean(psds, axis=0)

    if merge_grads:
        from ..channels.layout import _merge_grad_data
        psds = _merge_grad_data(psds)

    return plot_psds_topomap(
        psds=psds, freqs=freqs, pos=pos, agg_fun=agg_fun, vmin=vmin,
        vmax=vmax, bands=bands, cmap=cmap, dB=dB, normalize=normalize,
        cbar_fmt=cbar_fmt, outlines=outlines, show=show)


def plot_psds_topomap(
        psds, freqs, pos, agg_fun=None, vmin=None, vmax=None, bands=None,
        cmap='RdBu_r', dB=True, normalize=False, cbar_fmt='%0.3f',
        outlines='head', show=True):
    """Plot spatial maps of PSDs

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

    cmap : matplotlib colormap
        Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
        'Reds'.
    dB : bool
        If True, transform data to decibels (with ``10 * np.log10(data)``)
        following the application of `agg_fun`. Only valid if normalize is
        False.
    normalize : bool
        If True, each band will be devided by the total power. Defaults to
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

        _plot_topomap_multi_cbar(data, pos, ax, title=title,
                                 vmin=vmin, vmax=vmax, cmap=cmap,
                                 colorbar=True, unit=unit, cbar_fmt=cbar_fmt)
    tight_layout(fig=fig)
    fig.canvas.draw()
    if show:
        plt.show()
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
        picks = pick_types(tfr.info, meg=ch_type, ref_meg=False)
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
    itmax = min(itmax, len(tfr.times) - 1)
    ifmax = min(ifmax, len(tfr.freqs) - 1)
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
    plt.show()


def _find_peaks(evoked, npeaks):
    """Helper function for finding peaks from evoked data
    Returns ``npeaks`` biggest peaks as a list of time points.
    """
    argrelmax = _get_argrelmax()
    gfp = evoked.data.std(axis=0)
    order = len(evoked.times) // 30
    if order < 1:
        order = 1
    peaks = argrelmax(gfp, order=order, axis=0)[0]
    if len(peaks) > npeaks:
        max_indices = np.argsort(gfp[peaks])[-npeaks:]
        peaks = np.sort(peaks[max_indices])
    times = evoked.times[peaks]
    if len(times) == 0:
        times = [evoked.times[gfp.argmax()]]
    return times
