# -*- coding: utf-8 -*-
"""Intracranial elecrode localization GUI for finding contact locations."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from functools import partial

from matplotlib.colors import LinearSegmentedColormap

from PyQt5 import QtCore, QtGui, Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QGridLayout,
                             QVBoxLayout, QHBoxLayout, QLabel,
                             QMessageBox, QWidget,
                             QListView, QSlider, QPushButton,
                             QComboBox, QPlainTextEdit)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patheffects

from .._freesurfer import _check_subject_dir, _import_nibabel
from ..viz.backends.renderer import _get_renderer
from ..surface import _read_mri_surface, _voxel_neighbors
from ..transforms import (apply_trans, _frame_to_str, _get_trans,
                          invert_transform)
from ..utils import logger, _check_fname, _validate_type, verbose, warn

_IMG_LABELS = [['I', 'P'], ['I', 'L'], ['P', 'L']]
_CH_PLOT_SIZE = 1024
_ZOOM_STEP_SIZE = 5

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


class ComboBox(QComboBox):
    """Dropdown menu that emits a click when popped up."""

    clicked = QtCore.pyqtSignal()

    def showPopup(self):
        """Override show popup method to emit click."""
        self.clicked.emit()
        super(ComboBox, self).showPopup()


def _make_slice_plot(width=4, height=4, dpi=300):
    fig = Figure(figsize=(width, height))
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(bottom=0, left=0, right=1,
                        top=1, wspace=0, hspace=0)
    ax.set_facecolor('k')
    # clean up excess plot text, invert
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return canvas, fig


class IntracranialElectrodeLocator(QMainWindow):
    """Locate electrode contacts using a coregistered MRI and CT."""

    def __init__(self, info, trans, aligned_ct, subject=None,
                 subjects_dir=None, groups=None, verbose=None):
        """GUI for locating intracranial electrodes.

        .. note:: Images will be displayed using orientation information
                  obtained from the image header. Images will be resampled to
                  dimensions [256, 256, 256] for display.
        """
        # initialize QMainWindow class
        super(IntracranialElectrodeLocator, self).__init__()

        if not info.ch_names:
            raise ValueError('No channels found in `info` to locate')

        # store info for modification
        self._info = info
        self._verbose = verbose

        # load imaging data
        self._subject_dir = _check_subject_dir(subject, subjects_dir)
        self._load_image_data(aligned_ct)

        self._ch_alpha = 0.5
        self._radius = int(_CH_PLOT_SIZE // 100)  # starting 1/200 of image
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
            self._ras = np.array([0., 0., 0.])
        else:
            self._ras = self._chs[self._ch_names[self._ch_index]].copy()
        self._current_slice = apply_trans(
            self._ras_vox_t, self._ras).round().astype(int)
        self._group_channels(groups)

        # GUI design

        # Main plots: make one plot for each view; sagittal, coronal, axial
        plt_grid = QGridLayout()
        plts = [_make_slice_plot(), _make_slice_plot(), _make_slice_plot()]
        self._figs = [plts[0][1], plts[1][1], plts[2][1]]
        plt_grid.addWidget(plts[0][0], 0, 0)
        plt_grid.addWidget(plts[1][0], 0, 1)
        plt_grid.addWidget(plts[2][0], 1, 0)
        self._renderer = _get_renderer(
            name='IEEG Locator', size=(400, 400), bgcolor='w')
        # TODO: should eventually make sure the renderer here is actually
        # some PyVista(Qt) variant, not mayavi, otherwise the following
        # call will fail (hopefully it's rare that people who want to use this
        # have also set their MNE_3D_BACKEND=mayavi and/or don't have a working
        # pyvistaqt setup; also hopefully the refactoring to use the
        # Qt/notebook abstraction will make this easier, too):
        plt_grid.addWidget(self._renderer.plotter)

        # Channel selector
        self._ch_list = QListView()
        self._ch_list.setSelectionMode(Qt.QAbstractItemView.SingleSelection)
        self._ch_list.setMinimumWidth(150)
        self._set_ch_names()

        # Plots
        self._plot_images()

        # Menus
        button_hbox = self._get_button_bar()
        slider_hbox = self._get_slider_bar()
        bottom_hbox = self._get_bottom_bar()

        # Put everything together
        plot_ch_hbox = QHBoxLayout()
        plot_ch_hbox.addLayout(plt_grid)
        plot_ch_hbox.addWidget(self._ch_list)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(button_hbox)
        main_vbox.addLayout(slider_hbox)
        main_vbox.addLayout(plot_ch_hbox)
        main_vbox.addLayout(bottom_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

        # ready for user
        self._move_cursors_to_pos()
        self._ch_list.setFocus()  # always focus on list

    def _load_image_data(self, ct):
        """Get MRI and CT data to display and transforms to/from vox/RAS."""
        self._mri_data, self._vox_ras_t = _load_image(
            op.join(self._subject_dir, 'mri', 'brain.mgz'),
            'MRI Image', verbose=self._verbose)
        self._ras_vox_t = np.linalg.inv(self._vox_ras_t)

        self._voxel_sizes = np.array(self._mri_data.shape)
        self._img_ranges = [[0, self._voxel_sizes[1], 0, self._voxel_sizes[2]],
                            [0, self._voxel_sizes[0], 0, self._voxel_sizes[2]],
                            [0, self._voxel_sizes[0], 0, self._voxel_sizes[1]]]

        # ready ct
        self._ct_data, vox_ras_t = _load_image(ct, 'CT', verbose=self._verbose)
        if self._mri_data.shape != self._ct_data.shape or \
                not np.allclose(self._vox_ras_t, vox_ras_t, rtol=1e-6):
            raise ValueError('CT is not aligned to MRI, got '
                             f'CT shape={self._ct_data.shape}, '
                             f'MRI shape={self._mri_data.shape}, '
                             f'CT affine={vox_ras_t} and '
                             f'MRI affine={self._vox_ras_t}')

        if op.exists(op.join(self._subject_dir, 'surf', 'lh.seghead')):
            self._head = _read_mri_surface(
                op.join(self._subject_dir, 'surf', 'lh.seghead'))
            assert _frame_to_str[self._head['coord_frame']] == 'mri'
        else:
            warn('`seghead` not found, skipping head plot, see '
                 ':ref:`mne.bem.make_scalp_surfaces` to add the head')
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
                 'has been modified and these files have been deleted.')
            self._lh = self._rh = None

    def _make_ch_image(self, axis):
        """Make a plot to display the channel locations."""
        # Make channel data higher resolution so it looks better.
        ch_image = np.zeros((_CH_PLOT_SIZE, _CH_PLOT_SIZE)) * np.nan
        vx, vy, vz = self._voxel_sizes

        def color_ch_radius(ch_image, xf, yf, group, radius):
            # Take the fraction across each dimension of the RAS
            # coordinates converted to xyz and put a circle in that
            # position in this larger resolution image
            ex, ey = np.round(np.array([xf, yf]) * _CH_PLOT_SIZE).astype(int)
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (i**2 + j**2)**0.5 < radius:
                        # negative y because y axis is inverted
                        ch_image[-(ey + i), ex + j] = group
            return ch_image

        for name, ras in self._chs.items():
            # move from middle-centered (half coords positive, half negative)
            # to bottom-left corner centered (all coords positive).
            if np.isnan(ras).any():
                continue
            xyz = apply_trans(self._ras_vox_t, ras)
            # check if closest to that voxel
            dist = np.linalg.norm(xyz - self._current_slice)
            if dist < self._radius:
                x, y, z = xyz
                group = self._groups[name]
                r = self._radius - np.round(abs(dist)).astype(int)
                if axis == 0:
                    ch_image = color_ch_radius(
                        ch_image, y / vy, z / vz, group, r)
                elif axis == 1:
                    ch_image = color_ch_radius(
                        ch_image, x / vx, z / vx, group, r)
                elif axis == 2:
                    ch_image = color_ch_radius(
                        ch_image, x / vx, y / vy, group, r)
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
        self._images = dict(ct=list(), chs=list(),
                            cursor=list(), cursor2=list())
        ct_min, ct_max = np.nanmin(self._ct_data), np.nanmax(self._ct_data)
        text_kwargs = dict(fontsize='medium', weight='bold', color='#66CCEE',
                           family='monospace', ha='center', va='center',
                           path_effects=[patheffects.withStroke(
                               linewidth=4, foreground="k", alpha=0.75)])
        xyz = apply_trans(self._ras_vox_t, self._ras)
        for axis in range(3):
            ct_data = np.take(self._ct_data, self._current_slice[axis],
                              axis=axis).T
            self._images['ct'].append(self._figs[axis].axes[0].imshow(
                ct_data, cmap='gray', aspect='auto',
                vmin=ct_min, vmax=ct_max))
            self._images['chs'].append(
                self._figs[axis].axes[0].imshow(
                    self._make_ch_image(axis), aspect='auto',
                    extent=self._img_ranges[axis],
                    cmap=_CMAP, alpha=self._ch_alpha, vmin=0, vmax=_N_COLORS))
            self._images['cursor'].append(
                self._figs[axis].axes[0].plot(
                    (xyz[axis], xyz[axis]), (0, self._voxel_sizes[axis]),
                    color=[0, 1, 0], linewidth=1, alpha=0.5)[0])
            self._images['cursor2'].append(
                self._figs[axis].axes[0].plot(
                    (0, self._voxel_sizes[axis]), (xyz[axis], xyz[axis]),
                    color=[0, 1, 0], linewidth=1, alpha=0.5)[0])
            # label axes
            self._figs[axis].text(0.5, 0.05, _IMG_LABELS[axis][0],
                                  **text_kwargs)
            self._figs[axis].text(0.05, 0.5, _IMG_LABELS[axis][1],
                                  **text_kwargs)
            self._figs[axis].axes[0].axis(self._img_ranges[axis])
            self._figs[axis].canvas.mpl_connect(
                'scroll_event', self._on_scroll)
            self._figs[axis].canvas.mpl_connect(
                'button_release_event', partial(self._on_click, axis))
        # add head and brain in mm (convert from m)
        if self._head is not None:
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
        self._plot_3d_ch_pos()
        self._renderer.set_camera(azimuth=90, elevation=90, distance=300,
                                  focalpoint=tuple(self._ras))
        # update plots
        self._draw()
        self._renderer._update()

    def _scale_radius(self):
        """Scale the radius to mm."""
        shape = np.mean(self._ct_data.shape)  # this is Freesurfer shape (256)
        scale = np.diag(self._ras_vox_t)[:3].mean()
        return scale * self._radius * (shape / _CH_PLOT_SIZE)

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
            self._renderer.plotter.remove_actor(self._3d_chs.pop(name))
        if not any(np.isnan(self._chs[name])):
            radius = self._scale_radius()
            self._3d_chs[name] = self._renderer.sphere(
                tuple(self._chs[name]), scale=radius * 3,
                color=_UNIQUE_COLORS[self._groups[name] % _N_COLORS],
                opacity=self._ch_alpha)[0]
        if render:
            self._renderer._update()

    def _plot_3d_ch_pos(self, render=False):
        for name in self._chs:
            self._plot_3d_ch(name)
        if render:
            self._renderer._update()

    def _get_button_bar(self):
        """Make a bar with buttons for user interactions."""
        hbox = QHBoxLayout()

        help_button = QPushButton('Help')
        help_button.released.connect(self._show_help)
        hbox.addWidget(help_button)

        hbox.addStretch(8)

        hbox.addWidget(QLabel('Snap to Center'))
        self._snap_button = QPushButton('Off')
        self._snap_button.setMaximumWidth(25)  # not too big
        hbox.addWidget(self._snap_button)
        self._snap_button.released.connect(self._toggle_snap)
        self._toggle_snap()  # turn on to start

        hbox.addStretch(1)

        self._toggle_brain_button = QPushButton('Show Brain')
        self._toggle_brain_button.released.connect(self._toggle_show_brain)
        hbox.addWidget(self._toggle_brain_button)

        hbox.addStretch(1)

        mark_button = QPushButton('Mark')
        hbox.addWidget(mark_button)
        mark_button.released.connect(self._mark_ch)

        remove_button = QPushButton('Remove')
        hbox.addWidget(remove_button)
        remove_button.released.connect(self._remove_ch)

        self._group_selector = ComboBox()
        group_model = self._group_selector.model()

        for i in range(_N_COLORS):
            self._group_selector.addItem(' ')
            color = QtGui.QColor()
            color.setRgb(*(255 * np.array(_UNIQUE_COLORS[i % _N_COLORS])
                           ).round().astype(int))
            brush = QtGui.QBrush(color)
            brush.setStyle(QtCore.Qt.SolidPattern)
            group_model.setData(group_model.index(i, 0),
                                brush, QtCore.Qt.BackgroundRole)
        self._group_selector.clicked.connect(self._select_group)
        self._group_selector.currentIndexChanged.connect(
            self._select_group)
        hbox.addWidget(self._group_selector)

        # update background color for current selection
        self._update_group()

        return hbox

    def _get_slider_bar(self):
        """Make a bar with sliders on it."""

        def make_label(name):
            label = QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
            return label

        def make_slider(smin, smax, sval, sfun=None):
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(round(smin)))
            slider.setMaximum(int(round(smax)))
            slider.setValue(int(round(sval)))
            slider.setTracking(False)  # only update on release
            if sfun is not None:
                slider.valueChanged.connect(sfun)
            slider.keyPressEvent = self._key_press_event
            return slider

        slider_hbox = QHBoxLayout()

        ch_vbox = QVBoxLayout()
        ch_vbox.addWidget(make_label('ch alpha'))
        ch_vbox.addWidget(make_label('ch radius'))
        slider_hbox.addLayout(ch_vbox)

        ch_slider_vbox = QVBoxLayout()
        self._alpha_slider = make_slider(0, 100, self._ch_alpha * 100,
                                         self._update_ch_alpha)
        ch_plot_max = _CH_PLOT_SIZE // 50  # max 1 / 50 of plot size
        ch_slider_vbox.addWidget(self._alpha_slider)
        self._radius_slider = make_slider(0, ch_plot_max, self._radius,
                                          self._update_radius)
        ch_slider_vbox.addWidget(self._radius_slider)
        slider_hbox.addLayout(ch_slider_vbox)

        ct_vbox = QVBoxLayout()
        ct_vbox.addWidget(make_label('CT min'))
        ct_vbox.addWidget(make_label('CT max'))
        slider_hbox.addLayout(ct_vbox)

        ct_slider_vbox = QVBoxLayout()
        ct_min = int(round(np.nanmin(self._ct_data)))
        ct_max = int(round(np.nanmax(self._ct_data)))
        self._ct_min_slider = make_slider(
            ct_min, ct_max, ct_min, self._update_ct_scale)
        ct_slider_vbox.addWidget(self._ct_min_slider)
        self._ct_max_slider = make_slider(
            ct_min, ct_max, ct_max, self._update_ct_scale)
        ct_slider_vbox.addWidget(self._ct_max_slider)
        slider_hbox.addLayout(ct_slider_vbox)
        return slider_hbox

    def _get_bottom_bar(self):
        """Make a bar at the bottom with information in it."""
        hbox = QHBoxLayout()

        hbox.addStretch(10)

        self._intensity_label = QLabel('')  # update later
        hbox.addWidget(self._intensity_label)

        RAS_label = QLabel('RAS =')
        self._RAS_textbox = QPlainTextEdit('')  # update later
        self._RAS_textbox.setMaximumHeight(25)
        self._RAS_textbox.setMaximumWidth(200)
        self._RAS_textbox.focusOutEvent = self._update_RAS
        self._RAS_textbox.textChanged.connect(self._check_update_RAS)
        hbox.addWidget(RAS_label)
        hbox.addWidget(self._RAS_textbox)
        self._update_moved()  # update text now
        return hbox

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

    def _set_ch_names(self):
        """Add the channel names to the selector."""
        self._ch_list_model = QtGui.QStandardItemModel(self._ch_list)
        for name in self._ch_names:
            self._ch_list_model.appendRow(QtGui.QStandardItem(name))
            self._color_list_item(name=name)
        self._ch_list.setModel(self._ch_list_model)
        self._ch_list.clicked.connect(self._go_to_ch)
        self._ch_list.setCurrentIndex(
            self._ch_list_model.index(self._ch_index, 0))
        self._ch_list.keyPressEvent = self._key_press_event

    def _select_group(self):
        """Change the group label to the selection."""
        group = self._group_selector.currentIndex()
        self._groups[self._ch_names[self._ch_index]] = group
        # color differently if found already
        self._color_list_item(self._ch_names[self._ch_index])
        self._update_group()

    def _update_group(self):
        """Set background for closed group menu."""
        group = self._group_selector.currentIndex()
        rgb = (255 * np.array(_UNIQUE_COLORS[group % _N_COLORS])
               ).round().astype(int)
        self._group_selector.setStyleSheet(
            'background-color: rgb({:d},{:d},{:d})'.format(*rgb))
        self._group_selector.update()

    def _on_scroll(self, event):
        """Process mouse scroll wheel event to zoom."""
        self._zoom(event.step, draw=True)

    def _zoom(self, sign=1, draw=False):
        """Zoom in on the image."""
        delta = _ZOOM_STEP_SIZE * sign
        for axis, fig in enumerate(self._figs):
            xmid = self._images['cursor'][axis].get_xdata()[0]
            ymid = self._images['cursor2'][axis].get_ydata()[0]
            xmin, xmax = fig.axes[0].get_xlim()
            ymin, ymax = fig.axes[0].get_ylim()
            xwidth = (xmax - xmin) / 2 - delta
            ywidth = (ymax - ymin) / 2 - delta
            if xwidth <= 0 or ywidth <= 0:
                return
            fig.axes[0].set_xlim(xmid - xwidth, xmid + xwidth)
            fig.axes[0].set_ylim(ymid - ywidth, ymid + ywidth)
            self._images['cursor'][axis].set_ydata([ymin, ymax])
            self._images['cursor2'][axis].set_xdata([xmin, xmax])
            if draw:
                self._figs[axis].canvas.draw()

    def _update_ch_selection(self):
        """Update which channel is selected."""
        name = self._ch_names[self._ch_index]
        self._ch_list.setCurrentIndex(
            self._ch_list_model.index(self._ch_index, 0))
        self._group_selector.setCurrentIndex(self._groups[name])
        self._update_group()
        if not np.isnan(self._chs[name]).any():
            self._ras[:] = self._chs[name]
            self._move_cursors_to_pos()
            self._update_camera(render=True)
            self._draw()

    def _go_to_ch(self, index):
        """Change current channel to the item selected."""
        self._ch_index = index.row()
        self._update_ch_selection()

    @pyqtSlot()
    def _next_ch(self):
        """Increment the current channel selection index."""
        self._ch_index = (self._ch_index + 1) % len(self._ch_names)
        self._update_ch_selection()

    @pyqtSlot()
    def _update_RAS(self, event):
        """Interpret user input to the RAS textbox."""
        text = self._RAS_textbox.toPlainText().replace('\n', '')
        ras = text.split(',')
        if len(ras) != 3:
            ras = text.split(' ')  # spaces also okay as in freesurfer
        ras = [var.lstrip().rstrip() for var in ras]

        if len(ras) != 3:
            self._update_moved()  # resets RAS label
            return
        all_float = all([all([dig.isdigit() or dig in ('-', '.')
                              for dig in var]) for var in ras])
        if not all_float:
            self._update_moved()  # resets RAS label
            return

        ras = np.array([float(var) for var in ras])
        xyz = apply_trans(self._ras_vox_t, ras)
        wrong_size = any([var < 0 or var > n for var, n in
                          zip(xyz, self._voxel_sizes)])
        if wrong_size:
            self._update_moved()  # resets RAS label
            return

        # valid RAS position, update and move
        self._ras = ras
        self._move_cursors_to_pos()

    @pyqtSlot()
    def _check_update_RAS(self):
        """Check whether the RAS textbox is done being edited."""
        if '\n' in self._RAS_textbox.toPlainText():
            self._update_RAS(event=None)
            self._ch_list.setFocus()  # remove focus from text edit

    def _color_list_item(self, name=None):
        """Color the item in the view list for easy id of marked channels."""
        name = self._ch_names[self._ch_index] if name is None else name
        color = QtGui.QColor('white')
        if not np.isnan(self._chs[name]).any():
            group = self._groups[name]
            color.setRgb(*[int(c * 255) for c in
                           _UNIQUE_COLORS[int(group) % _N_COLORS]])
        brush = QtGui.QBrush(color)
        brush.setStyle(QtCore.Qt.SolidPattern)
        self._ch_list_model.setData(
            self._ch_list_model.index(self._ch_names.index(name), 0),
            brush, QtCore.Qt.BackgroundRole)
        # color text black
        color = QtGui.QColor('black')
        brush = QtGui.QBrush(color)
        brush.setStyle(QtCore.Qt.SolidPattern)
        self._ch_list_model.setData(
            self._ch_list_model.index(self._ch_names.index(name), 0),
            brush, QtCore.Qt.ForegroundRole)

    @pyqtSlot()
    def _toggle_snap(self):
        """Toggle snapping the contact location to the center of mass."""
        if self._snap_button.text() == 'Off':
            self._snap_button.setText('On')
            self._snap_button.setStyleSheet("background-color: green")
        else:  # text == 'On', turn off
            self._snap_button.setText('Off')
            self._snap_button.setStyleSheet("background-color: red")

    @pyqtSlot()
    def _mark_ch(self):
        """Mark the current channel as being located at the crosshair."""
        name = self._ch_names[self._ch_index]
        if self._snap_button.text() == 'Off':
            self._chs[name][:] = self._ras
        else:
            coord = apply_trans(self._ras_vox_t, self._ras.copy())
            shape = np.mean(self._mri_data.shape)  # Freesurfer shape (256)
            voxels_max = int(
                4 / 3 * np.pi * (shape * self._radius / _CH_PLOT_SIZE)**3)
            neighbors = _voxel_neighbors(
                coord, self._ct_data, thresh=0.5,
                voxels_max=voxels_max, use_relative=True)
            self._chs[name][:] = apply_trans(  # to surface RAS
                self._vox_ras_t, np.array(list(neighbors)).mean(axis=0))
        self._color_list_item()
        self._update_ch_images(draw=True)
        self._plot_3d_ch(name, render=True)
        self._save_ch_coords()
        self._next_ch()
        self._ch_list.setFocus()

    @pyqtSlot()
    def _remove_ch(self):
        """Remove the location data for the current channel."""
        name = self._ch_names[self._ch_index]
        self._chs[name] *= np.nan
        self._color_list_item()
        self._save_ch_coords()
        self._update_ch_images(draw=True)
        self._plot_3d_ch(name, render=True)
        self._next_ch()
        self._ch_list.setFocus()

    def _draw(self, axis=None):
        """Update the figures with a draw call."""
        for axis in (range(3) if axis is None else [axis]):
            self._figs[axis].canvas.draw()

    def _update_ch_images(self, axis=None, draw=False):
        """Update the channel image(s)."""
        for axis in range(3) if axis is None else [axis]:
            self._images['chs'][axis].set_data(
                self._make_ch_image(axis))
            if draw:
                self._draw(axis)

    def _update_ct_images(self, axis=None, draw=False):
        """Update the CT image(s)."""
        for axis in range(3) if axis is None else [axis]:
            ct_data = np.take(self._ct_data, self._current_slice[axis],
                              axis=axis).T
            # Threshold the CT so only bright objects (electrodes) are visible
            ct_data[ct_data < self._ct_min_slider.value()] = np.nan
            ct_data[ct_data > self._ct_max_slider.value()] = np.nan
            self._images['ct'][axis].set_data(ct_data)
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
        new_min = self._ct_min_slider.value()
        new_max = self._ct_max_slider.value()
        # handle inversions
        self._ct_min_slider.setValue(min([new_min, new_max]))
        self._ct_max_slider.setValue(max([new_min, new_max]))
        self._update_ct_images(draw=True)

    def _update_radius(self):
        """Update channel plot radius."""
        self._radius = np.round(self._radius_slider.value()).astype(int)
        self._update_ch_images(draw=True)
        self._plot_3d_ch_pos(render=True)
        self._ch_list.setFocus()  # remove focus from 3d plotter

    def _update_ch_alpha(self):
        """Update channel plot alpha."""
        self._ch_alpha = self._alpha_slider.value() / 100
        for axis in range(3):
            self._images['chs'][axis].set_alpha(self._ch_alpha)
        self._draw()
        self._plot_3d_ch_pos(render=True)
        self._ch_list.setFocus()  # remove focus from 3d plotter

    def _get_click_pos(self, axis, x, y):
        """Get which axis was clicked and where."""
        fx, fy = self._figs[axis].transFigure.inverted().transform((x, y))
        xmin, xmax = self._figs[axis].axes[0].get_xlim()
        ymin, ymax = self._figs[axis].axes[0].get_ylim()
        return (fx * (xmax - xmin) + xmin, fy * (ymax - ymin) + ymin)

    def _move_cursors_to_pos(self):
        """Move the cursors to a position."""
        x, y, z = apply_trans(self._ras_vox_t, self._ras)
        self._current_slice = np.array([x, y, z]).round().astype(int)
        self._move_cursor_to(0, x=y, y=z)
        self._move_cursor_to(1, x=x, y=z)
        self._move_cursor_to(2, x=x, y=y)
        self._zoom(0)  # doesn't actually zoom just resets view to center
        self._update_images(draw=True)
        self._update_moved()

    def _move_cursor_to(self, axis, x, y):
        """Move the cursors to a position for a given subplot."""
        self._images['cursor2'][axis].set_ydata([y, y])
        self._images['cursor'][axis].set_xdata([x, x])

    def _show_help(self):
        """Show the help menu."""
        QMessageBox.information(
            self, 'Help',
            "Help:\n'm': mark channel location\n"
            "'r': remove channel location\n"
            "'b': toggle viewing of brain in T1\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "page up/page down arrow: anterior/posterior")

    def _toggle_show_brain(self):
        """Toggle whether the brain/MRI is being shown."""
        if 'mri' in self._images:
            for img in self._images['mri']:
                img.remove()
            self._images.pop('mri')
            self._toggle_brain_button.setText('Show Brain')
        else:
            self._images['mri'] = list()
            for axis in range(3):
                mri_data = np.take(self._mri_data,
                                   self._current_slice[axis], axis=axis).T
                self._images['mri'].append(self._figs[axis].axes[0].imshow(
                    mri_data, cmap='hot', aspect='auto', alpha=0.25))
            self._toggle_brain_button.setText('Hide Brain')
        self._draw()

    def _key_press_event(self, event):
        """Execute functions when the user presses a key."""
        if event.key() == 'escape':
            self.close()

        if event.text() == 'h':
            self._show_help()

        if event.text() == 'm':
            self._mark_ch()

        if event.text() == 'r':
            self._remove_ch()

        if event.text() == 'b':
            self._toggle_show_brain()

        if event.text() in ('=', '+', '-'):
            self._zoom(sign=-2 * (event.text() == '-') + 1, draw=True)

        # Changing slices
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                           QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                           QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
            if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                self._ras[2] += 2 * (event.key() == QtCore.Qt.Key_Up) - 1
            elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                self._ras[0] += 2 * (event.key() == QtCore.Qt.Key_Right) - 1
            elif event.key() in (QtCore.Qt.Key_PageUp,
                                 QtCore.Qt.Key_PageDown):
                self._ras[1] += 2 * (event.key() == QtCore.Qt.Key_PageUp) - 1
            self._move_cursors_to_pos()

    def _on_click(self, axis, event):
        """Move to view on MRI and CT on click."""
        # Transform coordinates to figure coordinates
        pos = self._get_click_pos(axis, event.x, event.y)
        logger.info(f'Clicked axis {axis} at pos {pos}')

        if axis is not None and pos is not None:
            xyz = apply_trans(self._ras_vox_t, self._ras)
            if axis == 0:
                xyz[[1, 2]] = pos
            elif axis == 1:
                xyz[[0, 2]] = pos
            elif axis == 2:
                xyz[[0, 1]] = pos
            self._ras = apply_trans(self._vox_ras_t, xyz)
            self._move_cursors_to_pos()

    def _update_moved(self):
        """Update when cursor position changes."""
        self._RAS_textbox.setPlainText('{:.2f}, {:.2f}, {:.2f}'.format(
            *self._ras))
        self._intensity_label.setText('intensity = {:.2f}'.format(
            self._ct_data[tuple(self._current_slice)]))
