# -*- coding: utf-8 -*-
"""Intracranial elecrode localization GUI for finding contact locations."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from functools import partial
import platform

from scipy.ndimage import maximum_filter

from matplotlib import patheffects
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .._freesurfer import _import_nibabel
from ..viz.backends.renderer import _get_renderer
from ..viz.utils import safe_event
from ..surface import _read_mri_surface, _voxel_neighbors, _marching_cubes
from ..transforms import (apply_trans, _frame_to_str, _get_trans,
                          invert_transform)
from ..utils import logger, _check_fname, _validate_type, verbose, warn
from .. import pick_types

_IMG_LABELS = [['I', 'P'], ['I', 'L'], ['P', 'L']]
_CH_PLOT_SIZE = 1024
_ZOOM_STEP_SIZE = 5
_RADIUS_SCALAR = 0.4
_TUBE_SCALAR = 0.1
_BOLT_SCALAR = 30  # mm
_CH_MENU_WIDTH = 30 if platform.system() == 'Windows' else 10

# 20 colors generated to be evenly spaced in a cube, worked better than
# matplotlib color cycle
_UNIQUE_COLORS = [(0.1, 0.42, 0.43), (0.9, 0.34, 0.62), (0.47, 0.51, 0.3),
                  (0.47, 0.55, 0.99), (0.79, 0.68, 0.06), (0.34, 0.74, 0.05),
                  (0.58, 0.87, 0.13), (0.86, 0.98, 0.4), (0.92, 0.91, 0.66),
                  (0.77, 0.38, 0.34), (0.9, 0.37, 0.1), (0.2, 0.62, 0.9),
                  (0.22, 0.65, 0.64), (0.14, 0.94, 0.8), (0.34, 0.31, 0.68),
                  (0.59, 0.28, 0.74), (0.46, 0.19, 0.94), (0.37, 0.93, 0.7),
                  (0.56, 0.86, 0.55), (0.67, 0.69, 0.44)]
_N_COLORS = len(_UNIQUE_COLORS)
_CMAP = LinearSegmentedColormap.from_list(
    'ch_colors', _UNIQUE_COLORS, N=_N_COLORS)


def _load_image(img, name, verbose=True):
    """Load data from a 3D image file (e.g. CT, MR)."""
    nib = _import_nibabel('use iEEG GUI')
    if not isinstance(img, nib.spatialimages.SpatialImage):
        if verbose:
            logger.info(f'Loading {img}')
        _check_fname(img, overwrite='read', must_exist=True, name=name)
        img = nib.load(img)
    # get data
    orig_data = np.array(img.dataobj).astype(np.float32)
    # reorient data to RAS
    ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine)).astype(int)
    ras_ornt = nib.orientations.axcodes2ornt('RAS')
    ornt_trans = nib.orientations.ornt_transform(ornt, ras_ornt)
    img_data = nib.orientations.apply_orientation(orig_data, ornt_trans)
    orig_mgh = nib.MGHImage(orig_data, img.affine)
    aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, img.shape)
    vox_ras_t = np.dot(orig_mgh.header.get_vox2ras_tkr(), aff_trans)
    return img_data, vox_ras_t


def _make_slice_plot(width=4, height=4, dpi=300):
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_facecolor('k')
    # clean up excess plot text, invert
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return canvas, fig


class IntracranialElectrodeLocator():
    """Locate electrode contacts using a coregistered MRI and CT."""

    _xy_idx = (
        (1, 2),
        (0, 2),
        (0, 1),
    )

    def __init__(self, info, trans, aligned_ct, subject=None,
                 subjects_dir=None, groups=None, verbose=None):
        """GUI for locating intracranial electrodes.

        .. note:: Images will be displayed using orientation information
                  obtained from the image header. Images will be resampled to
                  dimensions [256, 256, 256] for display.
        """
        if not info.ch_names:
            raise ValueError('No channels found in `info` to locate')

        # store info for modification
        self._info = info
        self._seeg_idx = pick_types(self._info, meg=False, seeg=True)
        self._verbose = verbose

        # channel plotting default parameters
        self._ch_alpha = 0.5
        self._radius = int(_CH_PLOT_SIZE // 100)  # starting 1/100 of image

        # load imaging data
        self._subject_dir = op.join(subjects_dir, subject)
        self._load_image_data(aligned_ct)

        # initialize channel data
        self._ch_index = 0
        # load data, apply trans
        self._head_mri_t = _get_trans(trans, 'head', 'mri')[0]
        self._mri_head_t = invert_transform(self._head_mri_t)
        # load channels, convert from m to mm
        self._chs = {name: apply_trans(self._head_mri_t, ch['loc'][:3]) * 1000
                     for name, ch in zip(info.ch_names, info['chs'])}
        self._ch_names = list(self._chs.keys())
        # set current position
        if np.isnan(self._chs[self._ch_names[self._ch_index]]).any():
            ras = [0., 0., 0.]
        else:
            ras = self._chs[self._ch_names[self._ch_index]]
        self._set_ras(ras, update_plots=False)
        self._group_channels(groups)

        # GUI design
        self._configure_ui()

        # ready for user
        self._move_cursors_to_pos()
        self._renderer._list_view_focus(self._ch_list)  # always focus on list

    def _configure_ui(self):
        self._renderer = _get_renderer(
            name='IEEG Locator', size=(400, 400), bgcolor='w')
        self._window = self._renderer._window_create()

        # Main plots: make one plot for each view; sagittal, coronal, axial
        plts = [_make_slice_plot(), _make_slice_plot(), _make_slice_plot()]
        self._figs = [plts[0][1], plts[1][1], plts[2][1]]
        plt_grid = self._renderer._layout_create(orientation='grid')
        self._renderer._layout_add_widget(plt_grid, plts[0][0], row=0, col=0)
        self._renderer._layout_add_widget(plt_grid, plts[1][0], row=0, col=1)
        self._renderer._layout_add_widget(plt_grid, plts[2][0], row=1, col=0)
        self._renderer._layout_add_widget(plt_grid, self._renderer.plotter,
                                          row=1, col=1)

        # Channel selector
        self._ch_list = self._renderer._list_view_initialize(
            items=self._ch_names,
            idx=0,
            callback=self._go_to_ch,
        )
        for name in self._ch_names:
            self._color_list_item(name=name)
        self._renderer._keypress_initialize(self._ch_list)
        self._renderer._keypress_add('h', self._show_help)
        self._renderer._keypress_add('m', self._mark_ch)
        self._renderer._keypress_add('r', self._remove_ch)
        self._renderer._keypress_add('b', self._toggle_show_brain)
        self._renderer._keypress_add(
            '+', partial(self._zoom, sign=1, draw=True))
        self._renderer._keypress_add(
            '-', partial(self._zoom, sign=-1, draw=True))
        self._renderer._keypress_add(
            '=', partial(self._zoom, sign=1, draw=True))
        self._renderer._keypress_add('escape', self.close, key=True)
        self._renderer._keypress_add(
            'up', partial(self._adjust_ras, x=True), key=True)
        self._renderer._keypress_add(
            'down', partial(self._adjust_ras, x=False), key=True)
        self._renderer._keypress_add(
            'right', partial(self._adjust_ras, y=True), key=True)
        self._renderer._keypress_add(
            'left', partial(self._adjust_ras, y=False), key=True)
        self._renderer._keypress_add(
            'page_up', partial(self._adjust_ras, z=True), key=True)
        self._renderer._keypress_add(
            'page_down', partial(self._adjust_ras, z=False), key=True)
        self._renderer._keypress_add(
            'comma', partial(self._adjust_ras, z=True), key=True)
        self._renderer._keypress_add(
            'period', partial(self._adjust_ras, z=False), key=True)

        # Plots
        self._plot_images()

        # Menus
        self._configure_dialogs()
        self._configure_tool_bar()
        self._configure_dock()
        self._configure_status_bar()

        # Add lines
        self._lines = dict()
        self._lines_2D = dict()
        for group in set(self._groups.values()):
            self._update_lines(group)

        # Put everything together
        plot_layout = self._renderer._layout_create(orientation="horizontal")
        self._renderer._layout_add_widget(plot_layout, plt_grid)
        self._renderer._layout_add_widget(plot_layout, self._ch_list)

        main_layout = self._renderer._layout_create()
        self._renderer._layout_add_widget(main_layout, plot_layout)

        self._renderer._window_initialize(
            window=self._window, central_layout=main_layout,
            theme_support=False)

    def show(self):
        """Display the window."""
        self._window.show()

    def grab(self):
        """Render the widget into a pixmap."""
        return self._window.grab()

    def close(self):
        """Close all plots and cleanup data structure."""
        # XXX: this patch is necessary to close normally. Must be investigated.
        self._renderer._window = self._renderer.figure.plotter.app_window
        self._renderer.close()
        self._window.close()

    def _load_image_data(self, ct):
        """Get MRI and CT data to display and transforms to/from vox/RAS."""
        # allows recon-all not to be finished (T1 made in a few minutes)
        mri_img = 'brain' if op.isfile(op.join(
            self._subject_dir, 'mri', 'brain.mgz')) else 'T1'
        self._mri_data, self._vox_ras_t = _load_image(
            op.join(self._subject_dir, 'mri', f'{mri_img}.mgz'),
            'MRI Image', verbose=self._verbose)
        self._ras_vox_t = np.linalg.inv(self._vox_ras_t)

        self._voxel_sizes = np.array(self._mri_data.shape)
        # We need our extents to land the centers of each pixel on the voxel
        # number. This code assumes 1mm isotropic...
        img_delta = 0.5
        self._img_extents = list(
            [-img_delta, self._voxel_sizes[idx[0]] - img_delta,
             -img_delta, self._voxel_sizes[idx[1]] - img_delta]
            for idx in self._xy_idx)
        ch_deltas = list(img_delta * (self._voxel_sizes[ii] / _CH_PLOT_SIZE)
                         for ii in range(3))
        self._ch_extents = list(
            [-ch_delta, self._voxel_sizes[idx[0]] - ch_delta,
             -ch_delta, self._voxel_sizes[idx[1]] - ch_delta]
            for idx, ch_delta in zip(self._xy_idx, ch_deltas))

        # ready ct
        self._ct_data, vox_ras_t = _load_image(ct, 'CT', verbose=self._verbose)
        if self._mri_data.shape != self._ct_data.shape or \
                not np.allclose(self._vox_ras_t, vox_ras_t, rtol=1e-6):
            raise ValueError('CT is not aligned to MRI, got '
                             f'CT shape={self._ct_data.shape}, '
                             f'MRI shape={self._mri_data.shape}, '
                             f'CT affine={vox_ras_t} and '
                             f'MRI affine={self._vox_ras_t}')
        self._ct_maxima = None  # don't compute until turned on

        if op.exists(op.join(self._subject_dir, 'surf', 'lh.seghead')):
            self._head = _read_mri_surface(
                op.join(self._subject_dir, 'surf', 'lh.seghead'))
            assert _frame_to_str[self._head['coord_frame']] == 'mri'
        else:
            warn('`seghead` not found, using marching cubes on CT for '
                 'head plot, use :ref:`mne.bem.make_scalp_surfaces` '
                 'to add the scalp surface instead of skull from the CT')
            self._head = None
        if op.exists(op.join(self._subject_dir, 'surf', 'lh.pial')):
            self._lh = _read_mri_surface(
                op.join(self._subject_dir, 'surf', 'lh.pial'))
            assert _frame_to_str[self._lh['coord_frame']] == 'mri'
            self._rh = _read_mri_surface(
                op.join(self._subject_dir, 'surf', 'rh.pial'))
            assert _frame_to_str[self._rh['coord_frame']] == 'mri'
        else:
            warn('`pial` surface not found, skipping adding to 3D '
                 'plot. This indicates the Freesurfer recon-all '
                 'has not finished or has been modified and '
                 'these files have been deleted.')
            self._lh = self._rh = None

    def _make_ch_image(self, axis, proj=False):
        """Make a plot to display the channel locations."""
        # Make channel data higher resolution so it looks better.
        ch_image = np.zeros((_CH_PLOT_SIZE, _CH_PLOT_SIZE)) * np.nan
        vxyz = self._voxel_sizes

        def color_ch_radius(ch_image, xf, yf, group, radius):
            # Take the fraction across each dimension of the RAS
            # coordinates converted to xyz and put a circle in that
            # position in this larger resolution image
            ex, ey = np.round(np.array([xf, yf]) * _CH_PLOT_SIZE).astype(int)
            ii = np.arange(-radius, radius + 1)
            ii_sq = ii * ii
            idx = np.where(ii_sq + ii_sq[:, np.newaxis] < radius * radius)
            # negative y because y axis is inverted
            ch_image[-(ey + ii[idx[1]]), ex + ii[idx[0]]] = group
            return ch_image

        for name, ras in self._chs.items():
            # move from middle-centered (half coords positive, half negative)
            # to bottom-left corner centered (all coords positive).
            if np.isnan(ras).any():
                continue
            xyz = apply_trans(self._ras_vox_t, ras)
            # check if closest to that voxel
            dist = np.linalg.norm(xyz - self._current_slice)
            if proj or dist < self._radius:
                group = self._groups[name]
                r = self._radius if proj else \
                    self._radius - np.round(abs(dist)).astype(int)
                xf, yf = (xyz / vxyz)[list(self._xy_idx[axis])]
                ch_image = color_ch_radius(ch_image, xf, yf, group, r)
        return ch_image

    @verbose
    def _save_ch_coords(self, info=None, verbose=None):
        """Save the location of the electrode contacts."""
        logger.info('Saving channel positions to `info`')
        if info is None:
            info = self._info
        for name, ch in zip(info.ch_names, info['chs']):
            ch['loc'][:3] = apply_trans(
                self._mri_head_t, self._chs[name] / 1000)  # mm->m

    def _plot_images(self):
        """Use the MRI and CT to make plots."""
        # Plot sagittal (0), coronal (1) or axial (2) view
        self._images = dict(ct=list(), chs=list(), ct_bounds=list(),
                            cursor_v=list(), cursor_h=list())
        ct_min, ct_max = np.nanmin(self._ct_data), np.nanmax(self._ct_data)
        text_kwargs = dict(fontsize='medium', weight='bold', color='#66CCEE',
                           family='monospace', ha='center', va='center',
                           path_effects=[patheffects.withStroke(
                               linewidth=4, foreground="k", alpha=0.75)])
        xyz = apply_trans(self._ras_vox_t, self._ras)
        for axis in range(3):
            plot_x_idx, plot_y_idx = self._xy_idx[axis]
            fig = self._figs[axis]
            ax = fig.axes[0]
            ct_data = np.take(self._ct_data, self._current_slice[axis],
                              axis=axis).T
            self._images['ct'].append(ax.imshow(
                ct_data, cmap='gray', aspect='auto', zorder=1,
                vmin=ct_min, vmax=ct_max))
            img_extent = self._img_extents[axis]  # x0, x1, y0, y1
            w, h = np.diff(np.array(img_extent).reshape(2, 2), axis=1)[:, 0]
            self._images['ct_bounds'].append(Rectangle(
                img_extent[::2], w, h, edgecolor='w', facecolor='none',
                alpha=0.25, lw=0.5, zorder=1.5))
            ax.add_patch(self._images['ct_bounds'][-1])
            self._images['chs'].append(ax.imshow(
                self._make_ch_image(axis), aspect='auto',
                extent=self._ch_extents[axis], zorder=3,
                cmap=_CMAP, alpha=self._ch_alpha, vmin=0, vmax=_N_COLORS))
            v_x = (xyz[plot_x_idx],) * 2
            v_y = img_extent[2:4]
            self._images['cursor_v'].append(ax.plot(
                v_x, v_y, color='lime', linewidth=0.5, alpha=0.5, zorder=8)[0])
            h_y = (xyz[plot_y_idx],) * 2
            h_x = img_extent[0:2]
            self._images['cursor_h'].append(ax.plot(
                h_x, h_y, color='lime', linewidth=0.5, alpha=0.5, zorder=8)[0])
            # label axes
            self._figs[axis].text(0.5, 0.05, _IMG_LABELS[axis][0],
                                  **text_kwargs)
            self._figs[axis].text(0.05, 0.5, _IMG_LABELS[axis][1],
                                  **text_kwargs)
            self._figs[axis].axes[0].axis(img_extent)
            self._figs[axis].canvas.mpl_connect(
                'scroll_event', self._on_scroll)
            self._figs[axis].canvas.mpl_connect(
                'button_release_event', partial(self._on_click, axis=axis))
        # add head and brain in mm (convert from m)
        if self._head is None:
            logger.info('Using marching cubes on CT for the '
                        '3D visualization panel')
            rr, tris = _marching_cubes(np.where(
                self._ct_data < np.quantile(self._ct_data, 0.95), 0, 1),
                [1])[0]
            rr = apply_trans(self._vox_ras_t, rr)
            self._renderer.mesh(
                *rr.T, triangles=tris, color='gray', opacity=0.2,
                reset_camera=False, render=False)
        else:
            self._renderer.mesh(
                *self._head['rr'].T * 1000, triangles=self._head['tris'],
                color='gray', opacity=0.2, reset_camera=False, render=False)
        if self._lh is not None and self._rh is not None:
            self._renderer.mesh(
                *self._lh['rr'].T * 1000, triangles=self._lh['tris'],
                color='white', opacity=0.2, reset_camera=False, render=False)
            self._renderer.mesh(
                *self._rh['rr'].T * 1000, triangles=self._rh['tris'],
                color='white', opacity=0.2, reset_camera=False, render=False)
        self._3d_chs = dict()
        for name in self._chs:
            self._plot_3d_ch(name)
        self._renderer.set_camera(azimuth=90, elevation=90, distance=300,
                                  focalpoint=tuple(self._ras))
        # update plots
        self._draw()
        self._renderer._update()

    def _update_camera(self, render=False):
        """Update the camera position."""
        self._renderer.set_camera(
            # needs fix, distance moves when focal point updates
            distance=self._renderer.plotter.camera.distance * 0.9,
            focalpoint=tuple(self._ras),
            reset_camera=False)

    def _plot_3d_ch(self, name, render=False):
        """Plot a single 3D channel."""
        if name in self._3d_chs:
            self._renderer.plotter.remove_actor(
                self._3d_chs.pop(name), render=False)
        if not any(np.isnan(self._chs[name])):
            self._3d_chs[name] = self._renderer.sphere(
                tuple(self._chs[name]), scale=1,
                color=_CMAP(self._groups[name])[:3], opacity=self._ch_alpha)[0]
            # The actor scale is managed differently than the glyph scale
            # in order not to recreate objects, we use the actor scale
            self._3d_chs[name].SetOrigin(self._chs[name])
            self._3d_chs[name].SetScale(self._radius * _RADIUS_SCALAR)
        if render:
            self._renderer._update()

    def _configure_dialogs(self):
        from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING
        self._help_dialog = self._renderer._dialog_create(
            title="Help",
            text="Help:\n'm': mark channel location\n"
                 "'r': remove channel location\n"
                 "'b': toggle viewing of brain in T1\n"
                 "'+'/'-': zoom\nleft/right arrow: left/right\n"
                 "up/down arrow: superior/inferior\n"
                 "left angle bracket/right angle bracket: anterior/posterior",
            info_text="",
            callback=lambda x: None,
            icon="Information",
            buttons=["Ok"],
            modal=not MNE_3D_BACKEND_TESTING,
        )

    def _configure_tool_bar(self):
        """Make a bar with buttons for user interactions."""
        self._renderer._tool_bar_initialize(window=self._window)
        self._renderer._tool_bar_add_button(
            name='help',
            desc='Help',
            func=self._show_help,
            action=False,
        )
        self._renderer._tool_bar_add_spacer()
        self._renderer._tool_bar_add_label("Snap to Center")
        self._snap_button = self._renderer._tool_bar_add_button(
            name='snap_to_center',
            desc='Off',
            func=self._toggle_snap,
            action=False,
        )
        self._toggle_snap()  # turn on to start
        self._renderer._tool_bar_add_spacer()
        self._toggle_brain_button = self._renderer._tool_bar_add_button(
            name='show_brain',
            desc='Show Brain',
            func=self._toggle_show_brain,
            action=False,
        )
        self._renderer._tool_bar_add_spacer()
        self._renderer._tool_bar_add_button(
            name='mark',
            desc='Mark',
            func=self._mark_ch,
            action=False,
        )
        self._renderer._tool_bar_add_button(
            name='remove',
            desc='Remove',
            func=self._remove_ch,
            action=False,
        )
        colors = list()
        for i in range(_N_COLORS):
            color = (255 * np.array(_CMAP(i))).round().astype(int)
            colors.append(color)
        self._group_selector = self._renderer._tool_bar_add_combo_box(
            name="Select Group",
            value=0,
            rng=[' '] * _N_COLORS,
            callback=self._select_group,
            indexing=True,
            colors=colors,
        )
        # update background color for current selection
        self._update_group()

    def _configure_dock(self):
        """Make a bar with sliders on it."""
        self._renderer._dock_initialize(
            window=self._window, area="top", name="")
        dock_layout = self._renderer._dock_layout
        row1_layout = self._renderer._layout_create(orientation='horizontal')
        self._alpha_slider = self._renderer._dock_add_slider(
            name="ch alpha",
            value=self._ch_alpha * 100,
            rng=[0, 100],
            callback=self._update_ch_alpha,
            layout=row1_layout,
        )
        ct_min = int(round(np.nanmin(self._ct_data)))
        ct_max = int(round(np.nanmax(self._ct_data)))
        self._ct_min_slider = self._renderer._dock_add_slider(
            name="CT min",
            value=ct_min,
            rng=[ct_min, ct_max],
            callback=self._update_ct_scale,
            layout=row1_layout,
        )
        self._renderer._layout_add_widget(dock_layout, row1_layout)
        row2_layout = self._renderer._layout_create(orientation='horizontal')
        ch_plot_max = _CH_PLOT_SIZE // 50  # max 1 / 50 of plot size
        self._radius_slider = self._renderer._dock_add_slider(
            name="ch radius",
            value=self._radius,
            rng=[0, ch_plot_max],
            callback=self._update_radius,
            layout=row2_layout,
        )
        self._ct_max_slider = self._renderer._dock_add_slider(
            name="CT max",
            value=ct_max,
            rng=[ct_min, ct_max],
            callback=self._update_ct_scale,
            layout=row2_layout,
        )
        self._renderer._layout_add_widget(dock_layout, row2_layout)

    def _configure_status_bar(self):
        """Make a bar at the bottom with information in it."""
        self._renderer._status_bar_initialize(self._window)
        self._toggle_show_mip_button = \
            self._renderer._status_bar_add_button(
                name='Show Max Intensity Proj',
                callback=self._toggle_show_mip,
            )
        self._toggle_show_max_button = \
            self._renderer._status_bar_add_button(
                name='Show Maxima',
                callback=self._toggle_show_max,
            )
        self._intensity_label = self._renderer._status_bar_add_label('')
        self._renderer._status_bar_add_label('VOX = ')
        self._VOX_textbox = self._renderer._status_bar_add_text(
            value='', callback=self._update_VOX
        )
        self._renderer._status_bar_add_label('RAS =')
        self._RAS_textbox = self._renderer._status_bar_add_text(
            value='', callback=self._update_RAS,
        )
        self._update_moved()  # update text now

    def _group_channels(self, groups):
        """Automatically find a group based on the name of the channel."""
        if groups is not None:
            for name in self._ch_names:
                if name not in groups:
                    raise ValueError(f'{name} not found in ``groups``')
                _validate_type(groups[name], (float, int), f'groups[{name}]')
            self.groups = groups
        else:
            i = 0
            self._groups = dict()
            base_names = dict()
            for name in self._ch_names:
                # strip all numbers from the name
                base_name = ''.join([letter for letter in name if
                                     not letter.isdigit() and letter != ' '])
                if base_name in base_names:
                    # look up group number by base name
                    self._groups[name] = base_names[base_name]
                else:
                    self._groups[name] = i
                    base_names[base_name] = i
                    i += 1

    def _update_lines(self, group, only_2D=False):
        """Draw lines that connect the points in a group."""
        if group in self._lines_2D:  # remove existing 2D lines first
            for line in self._lines_2D[group]:
                line.remove()
            self._lines_2D.pop(group)
        if only_2D:  # if not in projection, don't add 2D lines
            if self._toggle_show_mip_button.get_text() == \
                    'Show Max Intensity Proj':
                return
        elif group in self._lines:  # if updating 3D, remove first
            self._renderer.plotter.remove_actor(
                self._lines[group], render=False)
        pos = np.array([
            self._chs[ch] for i, ch in enumerate(self._ch_names)
            if self._groups[ch] == group and i in self._seeg_idx and
            not np.isnan(self._chs[ch]).any()])
        if len(pos) < 2:  # not enough points for line
            return
        # first, the insertion will be the point farthest from the origin
        # brains are a longer posterior-anterior, scale for this (80%)
        insert_idx = np.argmax(np.linalg.norm(pos * np.array([1, 0.8, 1]),
                                              axis=1))
        # second, find the farthest point from the insertion
        target_idx = np.argmax(np.linalg.norm(pos[insert_idx] - pos, axis=1))
        # third, make a unit vector and to add to the insertion for the bolt
        elec_v = pos[insert_idx] - pos[target_idx]
        elec_v /= np.linalg.norm(elec_v)
        if not only_2D:
            self._lines[group] = self._renderer.tube(
                [pos[target_idx]], [pos[insert_idx] + elec_v * _BOLT_SCALAR],
                radius=self._radius * _TUBE_SCALAR, color=_CMAP(group)[:3])[0]
        if self._toggle_show_mip_button.get_text() == \
                'Hide Max Intensity Proj':
            # add 2D lines on each slice plot if in max intensity projection
            target_vox = apply_trans(self._ras_vox_t, pos[target_idx])
            insert_vox = apply_trans(self._ras_vox_t,
                                     pos[insert_idx] + elec_v * _BOLT_SCALAR)
            lines_2D = list()
            for axis in range(3):
                x, y = self._xy_idx[axis]
                lines_2D.append(self._figs[axis].axes[0].plot(
                    [target_vox[x], insert_vox[x]],
                    [target_vox[y], insert_vox[y]],
                    color=_CMAP(group), linewidth=0.25, zorder=7)[0])
            self._lines_2D[group] = lines_2D

    def _select_group(self, idx):
        """Change the group label to the selection."""
        self._groups[self._ch_names[self._ch_index]] = idx
        # color differently if found already
        self._color_list_item(self._ch_names[self._ch_index])
        self._update_group()

    def _update_group(self):
        """Set background for closed group menu."""
        group = self._group_selector.get_value()
        rgb = (255 * np.array(_CMAP(group))).round().astype(int)
        style = {'background-color': 'rgb({:d},{:d},{:d})'.format(*rgb)}
        self._group_selector.set_style(style)
        self._group_selector.update()

    def _on_scroll(self, event):
        """Process mouse scroll wheel event to zoom."""
        self._zoom(event.step, draw=True)

    def _zoom(self, sign=1, draw=False):
        """Zoom in on the image."""
        delta = _ZOOM_STEP_SIZE * sign
        for axis, fig in enumerate(self._figs):
            xmid = self._images['cursor_v'][axis].get_xdata()[0]
            ymid = self._images['cursor_h'][axis].get_ydata()[0]
            xmin, xmax = fig.axes[0].get_xlim()
            ymin, ymax = fig.axes[0].get_ylim()
            xwidth = (xmax - xmin) / 2 - delta
            ywidth = (ymax - ymin) / 2 - delta
            if xwidth <= 0 or ywidth <= 0:
                return
            fig.axes[0].set_xlim(xmid - xwidth, xmid + xwidth)
            fig.axes[0].set_ylim(ymid - ywidth, ymid + ywidth)
            if draw:
                self._figs[axis].canvas.draw()

    def _update_ch_selection(self):
        """Update which channel is selected."""
        name = self._ch_names[self._ch_index]
        self._renderer._list_view_set_index(self._ch_list, self._ch_index)
        self._group_selector.set_value(self._groups[name])
        self._update_group()
        if not np.isnan(self._chs[name]).any():
            self._set_ras(self._chs[name])
            self._update_camera(render=True)
            self._draw()

    def _go_to_ch(self, index):
        """Change current channel to the item selected."""
        self._ch_index = index.row()
        self._update_ch_selection()

    def _next_ch(self):
        """Increment the current channel selection index."""
        self._ch_index = (self._ch_index + 1) % len(self._ch_names)
        self._update_ch_selection()

    def _update_RAS(self, event):
        """Interpret user input to the RAS textbox."""
        text = self._RAS_textbox.get_value()
        ras = self._convert_text(text, 'ras')
        if ras is not None:
            self._set_ras(ras)

    def _update_VOX(self, event):
        """Interpret user input to the RAS textbox."""
        text = self._VOX_textbox.get_value()
        ras = self._convert_text(text, 'vox')
        if ras is not None:
            self._set_ras(ras)

    def _convert_text(self, text, text_kind):
        text = text.replace('\n', '')
        vals = text.split(',')
        if len(vals) != 3:
            vals = text.split(' ')  # spaces also okay as in freesurfer
        vals = [var.lstrip().rstrip() for var in vals]
        try:
            vals = np.array([float(var) for var in vals]).reshape(3)
        except Exception:
            self._update_moved()  # resets RAS label
            return
        if text_kind == 'vox':
            vox = vals
            ras = apply_trans(self._vox_ras_t, vox)
        else:
            assert text_kind == 'ras'
            ras = vals
            vox = apply_trans(self._ras_vox_t, ras)
        wrong_size = any(var < 0 or var > n - 1 for var, n in
                         zip(vox, self._voxel_sizes))
        if wrong_size:
            self._update_moved()  # resets RAS label
            return
        return ras

    @property
    def _ras(self):
        return self._ras_safe

    def _adjust_ras(self, x=None, y=None, z=None):
        ras = np.array(self._ras)
        if x is not None:
            ras[2] += 1 if x else -1
        if y is not None:
            ras[0] += 1 if y else -1
        if z is not None:
            ras[1] += 1 if z else -1
        self._set_ras(ras)

    def _set_ras(self, ras, update_plots=True):
        ras = np.asarray(ras, dtype=float)
        assert ras.shape == (3,)
        msg = ', '.join(f'{x:0.2f}' for x in ras)
        logger.debug(f'Trying RAS:  ({msg}) mm')
        # clip to valid
        vox = apply_trans(self._ras_vox_t, ras)
        vox = np.array([
            np.clip(d, 0, self._voxel_sizes[ii] - 1)
            for ii, d in enumerate(vox)])
        # transform back, make write-only
        self._ras_safe = apply_trans(self._vox_ras_t, vox)
        self._ras_safe.flags['WRITEABLE'] = False
        msg = ', '.join(f'{x:0.2f}' for x in self._ras_safe)
        logger.debug(f'Setting RAS: ({msg}) mm')
        if update_plots:
            self._move_cursors_to_pos()

    @property
    def _vox(self):
        return apply_trans(self._ras_vox_t, self._ras)

    @property
    def _current_slice(self):
        return self._vox.round().astype(int)

    def _check_update_RAS(self):
        """Check whether the RAS textbox is done being edited."""
        if '\n' in self._RAS_textbox.get_value():
            self._update_RAS(event=None)
            # remove focus from text edit
            self._renderer._list_view_focus(self._ch_list)

    def _check_update_VOX(self):
        """Check whether the VOX textbox is done being edited."""
        if '\n' in self._VOX_textbox.get_value():
            self._update_VOX(event=None)
            # remove focus from text edit
            self._renderer._list_view_focus(self._ch_list)

    def _color_list_item(self, name=None):
        """Color the item in the view list for easy id of marked channels."""
        name = self._ch_names[self._ch_index] if name is None else name
        color = 'white'
        if not np.isnan(self._chs[name]).any():
            group = self._groups[name]
            color = [int(c * 255) for c in _CMAP(group)]
        self._renderer._list_view_set_item_color(
            list_view=self._ch_list,
            idx=self._ch_names.index(name),
            color=color,
        )

    def _toggle_snap(self):
        """Toggle snapping the contact location to the center of mass."""
        style = dict()
        if self._snap_button.get_text() == 'Off':
            self._snap_button.set_text('On')
            style["background-color"] = "green"
        else:  # text == 'On', turn off
            self._snap_button.set_text('Off')
            style["background-color"] = "red"
        self._snap_button.set_style(style)

    def _mark_ch(self):
        """Mark the current channel as being located at the crosshair."""
        name = self._ch_names[self._ch_index]
        if self._snap_button.get_text() == 'Off':
            self._chs[name][:] = self._ras
        else:
            shape = np.mean(self._mri_data.shape)  # Freesurfer shape (256)
            voxels_max = int(
                4 / 3 * np.pi * (shape * self._radius / _CH_PLOT_SIZE)**3)
            neighbors = _voxel_neighbors(
                self._vox, self._ct_data, thresh=0.5,
                voxels_max=voxels_max, use_relative=True)
            self._chs[name][:] = apply_trans(  # to surface RAS
                self._vox_ras_t, np.array(list(neighbors)).mean(axis=0))
        self._color_list_item()
        self._update_lines(self._groups[name])
        self._update_ch_images(draw=True)
        self._plot_3d_ch(name, render=True)
        self._save_ch_coords()
        self._next_ch()
        self._renderer._list_view_focus(self._ch_list)

    def _remove_ch(self):
        """Remove the location data for the current channel."""
        name = self._ch_names[self._ch_index]
        self._chs[name] *= np.nan
        self._color_list_item()
        self._save_ch_coords()
        self._update_lines(self._groups[name])
        self._update_ch_images(draw=True)
        self._plot_3d_ch(name, render=True)
        self._next_ch()
        self._renderer._list_view_focus(self._ch_list)

    def _draw(self, axis=None):
        """Update the figures with a draw call."""
        for axis in (range(3) if axis is None else [axis]):
            self._figs[axis].canvas.draw()

    def _update_ch_images(self, axis=None, draw=False):
        """Update the channel image(s)."""
        for axis in range(3) if axis is None else [axis]:
            self._images['chs'][axis].set_data(
                self._make_ch_image(axis))
            if self._toggle_show_mip_button.get_text() == \
                    'Hide Max Intensity Proj':
                self._images['mip_chs'][axis].set_data(
                    self._make_ch_image(axis, proj=True))
            if draw:
                self._draw(axis)

    def _update_ct_images(self, axis=None, draw=False):
        """Update the CT image(s)."""
        for axis in range(3) if axis is None else [axis]:
            ct_data = np.take(self._ct_data, self._current_slice[axis],
                              axis=axis).T
            # Threshold the CT so only bright objects (electrodes) are visible
            ct_data[ct_data < self._ct_min_slider.get_value()] = np.nan
            ct_data[ct_data > self._ct_max_slider.get_value()] = np.nan
            self._images['ct'][axis].set_data(ct_data)
            if 'local_max' in self._images:
                ct_max_data = np.take(
                    self._ct_maxima, self._current_slice[axis], axis=axis).T
                self._images['local_max'][axis].set_data(ct_max_data)
            if draw:
                self._draw(axis)

    def _update_mri_images(self, axis=None, draw=False):
        """Update the CT image(s)."""
        if 'mri' in self._images:
            for axis in range(3) if axis is None else [axis]:
                self._images['mri'][axis].set_data(
                    np.take(self._mri_data, self._current_slice[axis],
                            axis=axis).T)
                if draw:
                    self._draw(axis)

    def _update_images(self, axis=None, draw=True):
        """Update CT and channel images when general changes happen."""
        self._update_ct_images(axis=axis)
        self._update_ch_images(axis=axis)
        self._update_mri_images(axis=axis)
        if draw:
            self._draw(axis)

    def _update_ct_scale(self):
        """Update CT min slider value."""
        new_min = self._ct_min_slider.get_value()
        new_max = self._ct_max_slider.get_value()
        # handle inversions
        self._ct_min_slider.set_value(min([new_min, new_max]))
        self._ct_max_slider.set_value(max([new_min, new_max]))
        self._update_ct_images(draw=True)

    def _update_radius(self):
        """Update channel plot radius."""
        self._radius = np.round(self._radius_slider.get_value()).astype(int)
        if self._toggle_show_max_button.get_text():
            self._update_ct_maxima()
            self._update_ct_images()
        else:
            self._ct_maxima = None  # signals ct max is out-of-date
        self._update_ch_images(draw=True)
        for name, actor in self._3d_chs.items():
            if not np.isnan(self._chs[name]).any():
                actor.SetOrigin(self._chs[name])
                actor.SetScale(self._radius * _RADIUS_SCALAR)
        self._renderer._update()
        # remove focus from 3d plotter
        self._renderer._list_view_focus(self._ch_list)

    def _update_ch_alpha(self):
        """Update channel plot alpha."""
        self._ch_alpha = self._alpha_slider.get_value() / 100
        for axis in range(3):
            self._images['chs'][axis].set_alpha(self._ch_alpha)
        self._draw()
        for actor in self._3d_chs.values():
            actor.GetProperty().SetOpacity(self._ch_alpha)
        self._renderer._update()
        # remove focus from 3d plotter
        self._renderer._list_view_focus(self._ch_list)

    def _move_cursors_to_pos(self):
        """Move the cursors to a position."""
        for axis in range(3):
            x, y = self._vox[list(self._xy_idx[axis])]
            self._images['cursor_v'][axis].set_xdata([x, x])
            self._images['cursor_h'][axis].set_ydata([y, y])
        self._zoom(0)  # doesn't actually zoom just resets view to center
        self._update_images(draw=True)
        self._update_moved()

    def _show_help(self):
        """Show the help menu."""
        self._help_dialog.show()

    def _update_ct_maxima(self):
        """Compute the maximum voxels based on the current radius."""
        self._ct_maxima = maximum_filter(
            self._ct_data, (self._radius,) * 3) == self._ct_data
        self._ct_maxima[self._ct_data <= np.median(self._ct_data)] = \
            False
        self._ct_maxima = np.where(self._ct_maxima, 1, np.nan)  # transparent

    def _toggle_show_mip(self):
        """Toggle whether the maximum-intensity projection is shown."""
        if self._toggle_show_mip_button.get_text() == \
                'Show Max Intensity Proj':
            self._toggle_show_mip_button.set_text('Hide Max Intensity Proj')
            self._images['mip'] = list()
            self._images['mip_chs'] = list()
            ct_min, ct_max = np.nanmin(self._ct_data), np.nanmax(self._ct_data)
            for axis in range(3):
                ct_mip_data = np.max(self._ct_data, axis=axis).T
                self._images['mip'].append(
                    self._figs[axis].axes[0].imshow(
                        ct_mip_data, cmap='gray', aspect='auto',
                        vmin=ct_min, vmax=ct_max, zorder=5))
                # add circles for each channel
                xs, ys, colors = list(), list(), list()
                for name, ras in self._chs.items():
                    xyz = self._vox
                    xs.append(xyz[self._xy_idx[axis][0]])
                    ys.append(xyz[self._xy_idx[axis][1]])
                    colors.append(_CMAP(self._groups[name]))
                self._images['mip_chs'].append(
                    self._figs[axis].axes[0].imshow(
                        self._make_ch_image(axis, proj=True), aspect='auto',
                        extent=self._ch_extents[axis], zorder=6,
                        cmap=_CMAP, alpha=1, vmin=0, vmax=_N_COLORS))
            for group in set(self._groups.values()):
                self._update_lines(group, only_2D=True)
        else:
            for img in self._images['mip'] + self._images['mip_chs']:
                img.remove()
            self._images.pop('mip')
            self._images.pop('mip_chs')
            self._toggle_show_mip_button.set_text('Show Max Intensity Proj')
            for group in set(self._groups.values()):  # remove lines
                self._update_lines(group, only_2D=True)
        self._draw()

    def _toggle_show_max(self):
        """Toggle whether to color local maxima differently."""
        if self._toggle_show_max_button.get_text() == 'Show Maxima':
            self._toggle_show_max_button.set_text('Hide Maxima')
            # happens on initiation or if the radius is changed with it off
            if self._ct_maxima is None:  # otherwise don't recompute
                self._update_ct_maxima()
            self._images['local_max'] = list()
            for axis in range(3):
                ct_max_data = np.take(self._ct_maxima,
                                      self._current_slice[axis], axis=axis).T
                self._images['local_max'].append(
                    self._figs[axis].axes[0].imshow(
                        ct_max_data, cmap='autumn', aspect='auto',
                        vmin=0, vmax=1, zorder=4))
        else:
            for img in self._images['local_max']:
                img.remove()
            self._images.pop('local_max')
            self._toggle_show_max_button.set_text('Show Maxima')
        self._draw()

    def _toggle_show_brain(self):
        """Toggle whether the brain/MRI is being shown."""
        if 'mri' in self._images:
            for img in self._images['mri']:
                img.remove()
            self._images.pop('mri')
            self._toggle_brain_button.set_text('Show Brain')
        else:
            self._images['mri'] = list()
            for axis in range(3):
                mri_data = np.take(self._mri_data,
                                   self._current_slice[axis], axis=axis).T
                self._images['mri'].append(self._figs[axis].axes[0].imshow(
                    mri_data, cmap='hot', aspect='auto', alpha=0.25, zorder=2))
            self._toggle_brain_button.set_text('Hide Brain')
        self._draw()

    def _on_click(self, event, axis):
        """Move to view on MRI and CT on click."""
        if event.inaxes is self._figs[axis].axes[0]:
            # Data coordinates are voxel coordinates
            pos = (event.xdata, event.ydata)
            logger.info(f'Clicked {"XYZ"[axis]} ({axis}) axis at pos {pos}')
            xyz = self._vox
            xyz[list(self._xy_idx[axis])] = pos
            logger.debug(f'Using voxel  {list(xyz)}')
            ras = apply_trans(self._vox_ras_t, xyz)
            self._set_ras(ras)

    def _update_moved(self):
        """Update when cursor position changes."""
        self._RAS_textbox.set_value('{:.2f}, {:.2f}, {:.2f}'.format(
            *self._ras))
        self._VOX_textbox.set_value('{:3d}, {:3d}, {:3d}'.format(
            *self._current_slice))
        self._intensity_label.set_value('intensity = {:.2f}'.format(
            self._ct_data[tuple(self._current_slice)]))

    @safe_event
    def closeEvent(self, event):
        """Clean up upon closing the window."""
        self._renderer.plotter.close()
        self.close()
