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

import numpy as np
from scipy import linalg

from ..baseline import rescale
from ..io.constants import FIFF
from ..io.pick import pick_types
from ..utils import _clean_names, deprecated
from .utils import tight_layout, _setup_vmin_vmax, DEFAULTS
from .utils import _prepare_trellis, _check_delayed_ssp
from .utils import _draw_proj_checkbox


def _prepare_topo_plot(obj, ch_type, layout):
    """"Aux Function"""
    info = copy.deepcopy(obj.info)
    if layout is None and ch_type is not 'eeg':
        from ..layouts.layout import find_layout
        layout = find_layout(info)
    elif layout == 'auto':
        layout = None

    info['ch_names'] = _clean_names(info['ch_names'])
    for ii, this_ch in enumerate(info['chs']):
        this_ch['ch_name'] = info['ch_names'][ii]

    # special case for merging grad channels
    if (ch_type == 'grad' and FIFF.FIFFV_COIL_VV_PLANAR_T1 in
            np.unique([ch['coil_type'] for ch in info['chs']])):
        from ..layouts.layout import _pair_grad_sensors
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
            chs = [info['chs'][i] for i in picks]
            from ..layouts.layout import _find_topomap_coords
            pos = _find_topomap_coords(chs, layout)
        else:
            names = [n.upper() for n in layout.names]
            pos = [layout.pos[names.index(info['ch_names'][k].upper())]
                   for k in picks]

    return picks, pos, merge_grads, info['ch_names']


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
        from ..layouts.layout import _merge_grad_data
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


def plot_projs_topomap(projs, layout=None, cmap='RdBu_r', sensors='k,',
                       colorbar=False, res=64, size=1, show=True,
                       outlines='head', contours=6, image_interp='bilinear'):
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
        format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : scalar
        Side length of the topomaps in inches (only applies when plotting
        multiple topomaps at a time).
    show : bool
        Show figures if True
    outlines : 'head' | dict | None
        The outlines to be drawn. If 'head', a head scheme will be drawn. If
        dict, each key refers to a tuple of x and y positions. The values in
        'mask_pos' will serve as image mask. If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.

    Returns
    -------
    fig : instance of matplotlib figure
        Figure distributing one image per channel across sensor topography.
    """
    import matplotlib.pyplot as plt

    if layout is None:
        from ..layouts import read_layout
        layout = read_layout('Vectorview-all')

    if not isinstance(layout, list):
        layout = [layout]

    n_projs = len(projs)
    nrows = math.floor(math.sqrt(n_projs))
    ncols = math.ceil(n_projs / nrows)

    fig = plt.gcf()
    fig.clear()
    for k, proj in enumerate(projs):

        ch_names = _clean_names(proj['data']['col_names'])
        data = proj['data']['data'].ravel()

        idx = []
        for l in layout:
            is_vv = l.kind.startswith('Vectorview')
            if is_vv:
                from ..layouts.layout import _pair_grad_sensors_from_ch_names
                grad_pairs = _pair_grad_sensors_from_ch_names(ch_names)
                if grad_pairs:
                    ch_names = [ch_names[i] for i in grad_pairs]

            idx = [l.names.index(c) for c in ch_names if c in l.names]
            if len(idx) == 0:
                continue

            pos = l.pos[idx]
            if is_vv and grad_pairs:
                from ..layouts.layout import _merge_grad_data
                shape = (len(idx) / 2, 2, -1)
                pos = pos.reshape(shape).mean(axis=1)
                data = _merge_grad_data(data[grad_pairs]).ravel()

            break

        ax = plt.subplot(nrows, ncols, k + 1)
        ax.set_title(proj['desc'][:10] + '...')
        if len(idx):
            plot_topomap(data, pos, vmax=None, cmap=cmap,
                         sensors=sensors, res=res, outlines=outlines,
                         contours=contours, image_interp=image_interp)
            if colorbar:
                plt.colorbar()
        else:
            raise RuntimeError('Cannot find a proper layout for projection %s'
                               % proj['desc'])
    fig = ax.get_figure()
    if show and plt.get_backend() != 'agg':
        fig.show()
    tight_layout(fig=fig)

    return fig


def _check_outlines(pos, outlines, head_scale=0.85):
    """Check or create outlines for topoplot
    """
    pos = np.asarray(pos)
    if outlines in ('head', None):
        radius = 0.5
        step = 2 * np.pi / 101
        l = np.arange(0, 2 * np.pi + step, step)
        head_x = np.cos(l) * radius
        head_y = np.sin(l) * radius
        nose_x = np.array([0.18, 0, -0.18]) * radius
        nose_y = np.array([radius - .004, radius * 1.15, radius - .004])
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                         .532, .510, .489])
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199])
        x, y = pos[:, :2].T
        x_range = np.abs(x.max() - x.min())
        y_range = np.abs(y.max() - y.min())

        # shift and scale the electrode positions
        pos[:, 0] = head_scale * ((pos[:, 0] - x.min()) / x_range - 0.5)
        pos[:, 1] = head_scale * ((pos[:, 1] - y.min()) / y_range - 0.5)

        # Define the outline of the head, ears and nose
        if outlines is not None:
            outlines = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                            ear_left=(ear_x,  ear_y),
                            ear_right=(-ear_x,  ear_y))
        else:
            outlines = dict()

        outlines['mask_pos'] = head_x, head_y
    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image'
                             'mask')
    else:
        raise ValueError('Invalid value for `outlines')

    return pos, outlines


def _inside_contour(pos, contour):
    """Aux function"""
    npos, ncnt = len(pos), len(contour)
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


def plot_topomap(data, pos, vmax=None, vmin=None, cmap='RdBu_r', sensors='k,',
                 res=64, axis=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', image_mask=None,
                 contours=6, image_interp='bilinear'):
    """Plot a topographic map as image

    Parameters
    ----------
    data : array, length = n_points
        The data values to plot.
    pos : array, shape = (n_points, 2)
        For each data point, the x and y coordinates.
    vmin : float | callable
        The value specfying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data).
    vmax : float | callable
        The value specfying the upper bound of the color range.
        If None, the maximum absolute value is used. If vmin is None,
        but vmax is not, defaults to np.min(data).
        If callable, the output equals vmax(data).
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
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
        Default (None) equals:
        dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0,
             markersize=4)
    outlines : 'head' | dict | None
        The outlines to be drawn. If 'head', a head scheme will be drawn. If
        dict, each key refers to a tuple of x and y positions. The values in
        'mask_pos' will serve as image mask. If None, nothing will be drawn.
        Defaults to 'head'.
    image_mask : ndarray of bool, shape (res, res) | None
        The image mask to cover the interpolated surface. If None, it will be
        computed from the outline.
    contour : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    if data.ndim > 1:
        err = ("Data needs to be array of shape (n_sensors,); got shape "
               "%s." % str(data.shape))
        raise ValueError(err)
    elif len(data) != len(pos):
        err = ("Data and pos need to be of same length. Got data of shape %s, "
               "pos of shape %s." % (str(), str()))

    axes = plt.gca()
    axes.set_frame_on(False)

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

    plt.xticks(())
    plt.yticks(())
    pos, outlines = _check_outlines(pos, outlines)
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]

    ax = axis if axis else plt.gca()
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
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0] * 1.01]),
                      np.max(np.r_[xlim[1], mask_[:, 0] * 1.01]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1] * 1.01]),
                      np.max(np.r_[ylim[1], mask_[:, 1] * 1.01]))

    # interpolate data
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _griddata(pos_x, pos_y, data, Xi, Yi)

    if outlines is None:
        _is_default_outlines = False
    elif isinstance(outlines, dict):
        _is_default_outlines = any([k.startswith('head') for k in outlines])

    if _is_default_outlines and image_mask is None:
        # prepare masking
        image_mask, pos = _make_image_mask(outlines, pos, res)

    if image_mask is not None and not _is_default_outlines:
        Zi[~image_mask] = np.nan

    if mask_params is None:
        mask_params = DEFAULTS['mask_params'].copy()
    elif isinstance(mask_params, dict):
        params = dict((k, v) for k, v in DEFAULTS['mask_params'].items()
                      if k not in mask_params)
        mask_params.update(params)
    else:
        raise ValueError('`mask_params` must be of dict-type '
                         'or None')

    # plot map and countour
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax),
                   interpolation=image_interp)
    # plot outline
    linewidth = mask_params['markeredgewidth']
    if isinstance(outlines, dict):
        for k, (x, y) in outlines.items():
            if 'mask' in k:
                continue
            ax.plot(x, y, color='k', linewidth=linewidth)

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
        # remove nose offset and tweak
        patch = patches.Circle((0.5, 0.4687), radius=.46,
                               clip_on=True,
                               transform=ax.transAxes)
        im.set_clip_path(patch)
        ax.set_clip_path(patch)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch)

    if sensors is True:
        sensors = 'k,'
    if sensors and mask is None:
        ax.plot(pos_x, pos_y, sensors)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], sensors)

    if show_names:
        if show_names is True:
            show_names = lambda x: x
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', size='x-small')

    plt.subplots_adjust(top=.95)

    return im, cont


def _make_image_mask(outlines, pos, res):
    """Aux function
    """

    mask_ = np.c_[outlines['mask_pos']]
    xmin, xmax = (np.min(np.r_[np.inf, mask_[:, 0]]),
                  np.max(np.r_[-np.inf, mask_[:, 0]]))
    ymin, ymax = (np.min(np.r_[np.inf, mask_[:, 1]]),
                  np.max(np.r_[-np.inf, mask_[:, 1]]))

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


@deprecated('`plot_ica_topomap` is deprecated and will be removed in '
            'MNE 1.0. Use `plot_ica_components` instead')
def plot_ica_topomap(ica, source_idx, ch_type='mag', res=64, layout=None,
                     vmax=None, cmap='RdBu_r', sensors='k,', colorbar=True,
                     show=True):
    """This functoin is deprecated

    See ``plot_ica_components``.
    """
    return plot_ica_components(ica, source_idx, ch_type, res, layout,
                               vmax, cmap, sensors, colorbar)


def plot_ica_components(ica, picks=None, ch_type='mag', res=64,
                        layout=None, vmin=None, vmax=None, cmap='RdBu_r',
                        sensors='k,', colorbar=False, title=None,
                        show=True, outlines='head', contours=6,
                        image_interp='bilinear'):
    """Project unmixing matrix on interpolated sensor topogrpahy.

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA solution.
    picks : int | array-like | None
        The indices of the sources to be plotted.
        If None all are plotted in batches of 20.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. For 'grad', the gradiometers are
        collected in pairs and the RMS for each pair is plotted.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout is
        inferred from the data.
    vmin : float | callable
        The value specfying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data).
    vmax : float | callable
        The value specfying the upper bound of the color range.
        If None, the maximum absolute value is used. If vmin is None,
        but vmax is not, defaults to np.min(data).
        If callable, the output equals vmax(data).
    cmap : matplotlib colormap
        Colormap.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib
        plot format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    res : int
        The resolution of the topomap image (n pixels along each side).
    show : bool
        Call pyplot.show() at the end.
    outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', a head scheme will be drawn.
            If dict, each key refers to a tuple of x and y positions. The
            values in 'mask_pos' will serve as image mask. If None,
            nothing will be drawn. defaults to 'head'.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure or list
        The figure object(s).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid import make_axes_locatable

    if picks is None:  # plot components by sets of 20
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

    data = np.dot(ica.mixing_matrix_[:, picks].T,
                  ica.pca_components_[:ica.n_components_])

    if ica.info is None:
        raise RuntimeError('The ICA\'s measurement info is missing. Please '
                           'fit the ICA or add the corresponding info object.')

    data_picks, pos, merge_grads, names = _prepare_topo_plot(ica, ch_type,
                                                             layout)
    pos, outlines = _check_outlines(pos, outlines)
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
        from ..layouts.layout import _merge_grad_data
    for ii, data_, ax in zip(picks, data, axes):
        ax.set_title('IC #%03d' % ii, fontsize=12)
        data_ = _merge_grad_data(data_) if merge_grads else data_
        vmin_, vmax_ = _setup_vmin_vmax(data_, vmin, vmax)
        im = plot_topomap(data_.flatten(), pos, vmin=vmin_, vmax=vmax_,
                          res=res, axis=ax, cmap=cmap, outlines=outlines,
                          image_mask=image_mask, contours=contours,
                          image_interp=image_interp)[0]
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
                     ch_type='mag', baseline=None, mode='mean', layout=None,
                     vmax=None, vmin=None, cmap='RdBu_r', sensors='k,',
                     colorbar=True, unit=None, res=64, size=2, format='%1.1e',
                     show_names=False, title=None, axes=None, show=True):
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
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. For 'grad', the gradiometers are
        collected in pairs and the RMS for each pair is plotted.
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
    vmin : float | callable
        The value specfying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data).
    vmax : float | callable
        The value specfying the upper bound of the color range.
        If None, the maximum absolute value is used. If vmin is None,
        but vmax is not, defaults to np.min(data).
        If callable, the output equals vmax(data).
    cmap : matplotlib colormap
        Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
        'Reds'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib
        plot format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    unit : str | None
        The unit of the channel type used for colorbar labels.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : float
        Side length per topomap in inches.
    format : str
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
        Call pyplot.show() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the topography.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    picks, pos, merge_grads, names = _prepare_topo_plot(tfr, ch_type,
                                                        layout)
    if not show_names:
        names = None

    data = tfr.data

    if mode is not None and baseline is not None:
        data = rescale(data, tfr.times, baseline, mode, copy=True)

    # crop time
    itmin, itmax = None, None
    if tmin is not None:
        itmin = np.where(tfr.times >= tmin)[0][0]
    if tmax is not None:
        itmax = np.where(tfr.times <= tmax)[0][-1]

    # crop freqs
    ifmin, ifmax = None, None
    if fmin is not None:
        ifmin = np.where(tfr.freqs >= fmin)[0][0]
    if fmax is not None:
        ifmax = np.where(tfr.freqs <= fmax)[0][-1]

    data = data[picks, ifmin:ifmax, itmin:itmax]
    data = np.mean(np.mean(data, axis=2), axis=1)[:, np.newaxis]

    if merge_grads:
        from ..layouts.layout import _merge_grad_data
        data = _merge_grad_data(data)

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

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

    im, _ = plot_topomap(data[:, 0], pos, vmin=vmin, vmax=vmax,
                         axis=ax, cmap=cmap, image_interp='bilinear',
                         contours=False, names=names)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%3.2f', cmap=cmap)
        cbar.set_ticks((vmin, vmax))
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title('AU')

    if show:
        plt.show()

    return fig


def plot_evoked_topomap(evoked, times=None, ch_type='mag', layout=None,
                        vmax=None, vmin=None, cmap='RdBu_r', sensors='k,',
                        colorbar=True, scale=None, scale_time=1e3, unit=None,
                        res=64, size=1, format='%3.1f',
                        time_format='%01d ms', proj=False, show=True,
                        show_names=False, title=None, mask=None,
                        mask_params=None, outlines='head', contours=6,
                        image_interp='bilinear'):
    """Plot topographic maps of specific time points of evoked data

    Parameters
    ----------
    evoked : Evoked
        The Evoked object.
    times : float | array of floats | None.
        The time point(s) to plot. If None, 10 topographies will be shown
        will a regular time spacing between the first and last time instant.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg'
        The channel type to plot. For 'grad', the gradiometers are collected in
        pairs and the RMS for each pair is plotted.
    layout : None | Layout
        Layout instance specifying sensor positions (does not need to
        be specified for Neuromag data). If possible, the correct layout file
        is inferred from the data; if no appropriate layout file was found, the
        layout is automatically generated from the sensor locations.
    vmin : float | callable
        The value specfying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data).
    vmax : float | callable
        The value specfying the upper bound of the color range.
        If None, the maximum absolute value is used. If vmin is None,
        but vmax is not, defaults to np.min(data).
        If callable, the output equals vmax(data).
    cmap : matplotlib colormap
        Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
        'Reds'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses).
    colorbar : bool
        Plot a colorbar.
    scale : float | None
        Scale the data for plotting. If None, defaults to 1e6 for eeg, 1e13
        for grad and 1e15 for mag.
    scale_time : float | None
        Scale the time labels. Defaults to 1e3 (ms).
    unit : str | None
        The unit of the channel type used for colorbar label. If
        scale is None the unit is automatically determined.
    res : int
        The resolution of the topomap image (n pixels along each side).
    size : float
        Side length per topomap in inches.
    format : str
        String format for colorbar values.
    time_format : str
        String format for topomap values. Defaults to "%01d ms"
    proj : bool | 'interactive'
        If true SSP projections are applied before display. If 'interactive',
        a check box for reversible selection of SSP projection vectors will
        be show.
    show : bool
        Call pyplot.show() at the end.
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
        Default (None) equals:
        dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0,
             markersize=4)
    outlines : 'head' | dict | None
        The outlines to be drawn. If 'head', a head scheme will be drawn. If
        dict, each key refers to a tuple of x and y positions. The values in
        'mask_pos' will serve as image mask. If None, nothing will be drawn.
        Defaults to 'head'.
    contours : int | False | None
        The number of contour lines to draw. If 0, no contours will be drawn.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    """
    import matplotlib.pyplot as plt

    if ch_type.startswith('planar'):
        key = 'grad'
    else:
        key = ch_type

    if scale is None:
        scale = DEFAULTS['scalings'][key]
        unit = DEFAULTS['units'][key]

    if mask_params is None:
        mask_params = DEFAULTS['mask_params'].copy()
        mask_params['markersize'] *= size / 2.
        mask_params['markeredgewidth'] *= size / 2.

    if times is None:
        times = np.linspace(evoked.times[0], evoked.times[-1], 10)
    elif np.isscalar(times):
        times = [times]
    if len(times) > 20:
        raise RuntimeError('Too many plots requested. Please pass fewer '
                           'than 20 time instants.')
    tmin, tmax = evoked.times[[0, -1]]
    for t in times:
        if not tmin <= t <= tmax:
            raise ValueError('Times should be between %0.3f and %0.3f. (Got '
                             '%0.3f).' % (tmin, tmax, t))

    picks, pos, merge_grads, names = _prepare_topo_plot(evoked, ch_type,
                                                        layout)
    if not show_names:
        names = None

    n = len(times)
    nax = n + bool(colorbar)
    width = size * nax
    height = size * 1. + max(0, 0.1 * (4 - size))
    fig = plt.figure(figsize=(width, height))
    w_frame = plt.rcParams['figure.subplot.wspace'] / (2 * nax)
    top_frame = max((0.05 if title is None else 0.15), .2 / size)
    fig.subplots_adjust(left=w_frame, right=1 - w_frame, bottom=0,
                        top=1 - top_frame)
    time_idx = [np.where(evoked.times >= t)[0][0] for t in times]

    if proj is True and evoked.proj is not True:
        data = evoked.copy().apply_proj().data
    else:
        data = evoked.data

    data = data[np.ix_(picks, time_idx)] * scale
    if merge_grads:
        from ..layouts.layout import _merge_grad_data
        data = _merge_grad_data(data)

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)

    images, contours_ = [], []

    if mask is not None:
        _picks = picks[::2 if ch_type not in ['mag', 'eeg'] else 1]
        mask_ = mask[np.ix_(_picks, time_idx)]

    pos, outlines = _check_outlines(pos, outlines)
    if outlines is not None:
        image_mask, pos = _make_image_mask(outlines, pos, res)
    else:
        image_mask = None

    for i, t in enumerate(times):
        ax = plt.subplot(1, nax, i + 1)
        tp, cn = plot_topomap(data[:, i], pos, vmin=vmin, vmax=vmax,
                              sensors=sensors, res=res, names=names,
                              show_names=show_names, cmap=cmap,
                              mask=mask_[:, i] if mask is not None else None,
                              mask_params=mask_params, axis=ax,
                              outlines=outlines, image_mask=image_mask,
                              contours=contours, image_interp=image_interp)
        images.append(tp)
        if cn is not None:
            contours_.append(cn)
        if time_format is not None:
            plt.title(time_format % (t * scale_time))

    if colorbar:
        cax = plt.subplot(1, n + 1, n + 1)
        plt.colorbar(images[-1], ax=cax, cax=cax, ticks=[vmin, 0, vmax],
                     format=format)
        # resize the colorbar (by default the color fills the whole axes)
        cpos = cax.get_position()
        if size <= 1:
            cpos.x0 = 1 - (.7 + .1 / size) / nax
        cpos.x1 = cpos.x0 + .1 / nax
        cpos.y0 = .1
        cpos.y1 = .7
        cax.set_position(cpos)
        if unit is not None:
            cax.set_title(unit)

    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
                      picks=picks, images=images, contours=contours_,
                      time_idx=time_idx, scale=scale, merge_grads=merge_grads,
                      res=res, pos=pos, image_mask=image_mask,
                      plot_update_proj_callback=_plot_update_evoked_topomap)
        _draw_proj_checkbox(None, params)

    if title is not None:
        plt.suptitle(title, verticalalignment='top', size='x-large')
        tight_layout(pad=2 * size / 2.0, fig=fig)
    if show:
        plt.show()

    return fig
