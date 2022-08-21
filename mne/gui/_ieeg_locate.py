# -*- coding: utf-8 -*-
"""Intracranial elecrode localization GUI for finding contact locations."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import numpy as np
import platform

from scipy.ndimage import maximum_filter

from qtpy import QtCore, QtGui
from qtpy.QtCore import Slot, Signal
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                            QMessageBox, QWidget, QAbstractItemView,
                            QListView, QSlider, QPushButton,
                            QComboBox)

from matplotlib.colors import LinearSegmentedColormap

from ._core import SliceBrowser
from ..surface import _voxel_neighbors
from ..transforms import apply_trans, _get_trans, invert_transform
from ..utils import logger, _validate_type, verbose
from .. import pick_types

_CH_PLOT_SIZE = 1024
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


class ComboBox(QComboBox):
    """Dropdown menu that emits a click when popped up."""

    clicked = Signal()

    def showPopup(self):
        """Override show popup method to emit click."""
        self.clicked.emit()
        super(ComboBox, self).showPopup()


class IntracranialElectrodeLocator(SliceBrowser):
    """Locate electrode contacts using a coregistered MRI and CT."""

    def __init__(self, info, trans, aligned_ct, subject=None,
                 subjects_dir=None, groups=None, show=True, verbose=None):
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

        # initialize channel data
        self._ch_index = 0
        # load data, apply trans
        self._head_mri_t = _get_trans(trans, 'head', 'mri')[0]
        self._mri_head_t = invert_transform(self._head_mri_t)
        # load channels, convert from m to mm
        self._chs = {name: apply_trans(self._head_mri_t, ch['loc'][:3]) * 1000
                     for name, ch in zip(info.ch_names, info['chs'])}
        self._ch_names = list(self._chs.keys())
        self._group_channels(groups)

        # Initialize GUI
        super(IntracranialElectrodeLocator, self).__init__(
            base_image=aligned_ct, subject=subject, subjects_dir=subjects_dir)

        # set current position as current contact location if exists
        if not np.isnan(self._chs[self._ch_names[self._ch_index]]).any():
            self._set_ras(self._chs[self._ch_names[self._ch_index]],
                          update_plots=False)

        # add plots of contacts on top
        self._plot_ch_images()

        # Add lines
        self._lines = dict()
        self._lines_2D = dict()
        for group in set(self._groups.values()):
            self._update_lines(group)

        # ready for user
        self._move_cursors_to_pos()
        self._ch_list.setFocus()  # always focus on list

        if show:
            self.show()

    def _configure_ui(self):
        # data is loaded for an abstract base image, associate with ct
        self._ct_data = self._base_data
        self._images['ct'] = self._images['base']
        self._ct_maxima = None  # don't compute until turned on

        toolbar = self._configure_toolbar()
        slider_bar = self._configure_sliders()
        status_bar = self._configure_status_bar()
        self._ch_list = self._configure_channel_sidebar()  # need for updating

        plot_layout = QHBoxLayout()
        plot_layout.addLayout(self._plt_grid)
        plot_layout.addWidget(self._ch_list)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(toolbar)
        main_vbox.addLayout(slider_bar)
        main_vbox.addLayout(plot_layout)
        main_vbox.addLayout(status_bar)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

    def _configure_channel_sidebar(self):
        """Configure the sidebar to select channels/contacts."""
        ch_list = QListView()
        ch_list.setSelectionMode(QAbstractItemView.SingleSelection)
        max_ch_name_len = max([len(name) for name in self._chs])
        ch_list.setMinimumWidth(max_ch_name_len * _CH_MENU_WIDTH)
        ch_list.setMaximumWidth(max_ch_name_len * _CH_MENU_WIDTH)
        self._ch_list_model = QtGui.QStandardItemModel(ch_list)
        for name in self._ch_names:
            self._ch_list_model.appendRow(QtGui.QStandardItem(name))
            self._color_list_item(name=name)
        ch_list.setModel(self._ch_list_model)
        ch_list.clicked.connect(self._go_to_ch)
        ch_list.setCurrentIndex(
            self._ch_list_model.index(self._ch_index, 0))
        ch_list.keyPressEvent = self._key_press_event
        return ch_list

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
        with info._unlock():
            for name, ch in zip(info.ch_names, info['chs']):
                loc = ch['loc'].copy()
                loc[:3] = apply_trans(
                    self._mri_head_t, self._chs[name] / 1000)  # mm->m
                ch['loc'] = loc

    def _plot_ch_images(self):
        img_delta = 0.5
        ch_deltas = list(img_delta * (self._voxel_sizes[ii] / _CH_PLOT_SIZE)
                         for ii in range(3))
        self._ch_extents = list(
            [-ch_delta, self._voxel_sizes[idx[0]] - ch_delta,
             -ch_delta, self._voxel_sizes[idx[1]] - ch_delta]
            for idx, ch_delta in zip(self._xy_idx, ch_deltas))
        self._images['chs'] = list()
        for axis in range(3):
            fig = self._figs[axis]
            ax = fig.axes[0]
            self._images['chs'].append(ax.imshow(
                self._make_ch_image(axis), aspect='auto',
                extent=self._ch_extents[axis], zorder=3,
                cmap=_CMAP, alpha=self._ch_alpha, vmin=0, vmax=_N_COLORS))
        self._3d_chs = dict()
        for name in self._chs:
            self._plot_3d_ch(name)

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

    def _configure_toolbar(self):
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
            color.setRgb(*(255 * np.array(_CMAP(i))).round().astype(int))
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

    def _configure_sliders(self):
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

    def _configure_status_bar(self, hbox=None):
        hbox = QHBoxLayout() if hbox is None else hbox

        hbox.addStretch(3)

        self._toggle_show_mip_button = QPushButton('Show Max Intensity Proj')
        self._toggle_show_mip_button.released.connect(
            self._toggle_show_mip)
        hbox.addWidget(self._toggle_show_mip_button)

        self._toggle_show_max_button = QPushButton('Show Maxima')
        self._toggle_show_max_button.released.connect(
            self._toggle_show_max)
        hbox.addWidget(self._toggle_show_max_button)

        self._intensity_label = QLabel('')  # update later
        hbox.addWidget(self._intensity_label)

        # add SliceBrowser navigation items
        super(IntracranialElectrodeLocator, self)._configure_status_bar(
            hbox=hbox)
        return hbox

    def _move_cursors_to_pos(self):
        super(IntracranialElectrodeLocator, self)._move_cursors_to_pos()
        self._ch_list.setFocus()  # remove focus from text edit

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
            if self._toggle_show_mip_button.text() == \
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
        if self._toggle_show_mip_button.text() == 'Hide Max Intensity Proj':
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
        rgb = (255 * np.array(_CMAP(group))).round().astype(int)
        self._group_selector.setStyleSheet(
            'background-color: rgb({:d},{:d},{:d})'.format(*rgb))
        self._group_selector.update()

    def _update_ch_selection(self):
        """Update which channel is selected."""
        name = self._ch_names[self._ch_index]
        self._ch_list.setCurrentIndex(
            self._ch_list_model.index(self._ch_index, 0))
        self._group_selector.setCurrentIndex(self._groups[name])
        self._update_group()
        if not np.isnan(self._chs[name]).any():
            self._set_ras(self._chs[name])
            self._update_camera(render=True)
            self._draw()

    def _go_to_ch(self, index):
        """Change current channel to the item selected."""
        self._ch_index = index.row()
        self._update_ch_selection()

    @Slot()
    def _next_ch(self):
        """Increment the current channel selection index."""
        self._ch_index = (self._ch_index + 1) % len(self._ch_names)
        self._update_ch_selection()

    def _color_list_item(self, name=None):
        """Color the item in the view list for easy id of marked channels."""
        name = self._ch_names[self._ch_index] if name is None else name
        color = QtGui.QColor('white')
        if not np.isnan(self._chs[name]).any():
            group = self._groups[name]
            color.setRgb(*[int(c * 255) for c in _CMAP(group)])
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

    @Slot()
    def _toggle_snap(self):
        """Toggle snapping the contact location to the center of mass."""
        if self._snap_button.text() == 'Off':
            self._snap_button.setText('On')
            self._snap_button.setStyleSheet("background-color: green")
        else:  # text == 'On', turn off
            self._snap_button.setText('Off')
            self._snap_button.setStyleSheet("background-color: red")

    @Slot()
    def _mark_ch(self):
        """Mark the current channel as being located at the crosshair."""
        name = self._ch_names[self._ch_index]
        if self._snap_button.text() == 'Off':
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
        self._ch_list.setFocus()

    @Slot()
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
        self._ch_list.setFocus()

    def _update_ch_images(self, axis=None, draw=False):
        """Update the channel image(s)."""
        for axis in range(3) if axis is None else [axis]:
            self._images['chs'][axis].set_data(
                self._make_ch_image(axis))
            if self._toggle_show_mip_button.text() == \
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
            ct_data[ct_data < self._ct_min_slider.value()] = np.nan
            ct_data[ct_data > self._ct_max_slider.value()] = np.nan
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
        self._update_ch_images(axis=axis)
        self._update_mri_images(axis=axis)
        super()._update_images()

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
        if self._toggle_show_max_button.text() == 'Hide Maxima':
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
        self._ch_list.setFocus()  # remove focus from 3d plotter

    def _update_ch_alpha(self):
        """Update channel plot alpha."""
        self._ch_alpha = self._alpha_slider.value() / 100
        for axis in range(3):
            self._images['chs'][axis].set_alpha(self._ch_alpha)
        self._draw()
        for actor in self._3d_chs.values():
            actor.GetProperty().SetOpacity(self._ch_alpha)
        self._renderer._update()
        self._ch_list.setFocus()  # remove focus from 3d plotter

    def _show_help(self):
        """Show the help menu."""
        QMessageBox.information(
            self, 'Help',
            "Help:\n'm': mark channel location\n"
            "'r': remove channel location\n"
            "'b': toggle viewing of brain in T1\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "left angle bracket/right angle bracket: anterior/posterior")

    def _update_ct_maxima(self):
        """Compute the maximum voxels based on the current radius."""
        self._ct_maxima = maximum_filter(
            self._ct_data, (self._radius,) * 3) == self._ct_data
        self._ct_maxima[self._ct_data <= np.median(self._ct_data)] = \
            False
        self._ct_maxima = np.where(self._ct_maxima, 1, np.nan)  # transparent

    def _toggle_show_mip(self):
        """Toggle whether the maximum-intensity projection is shown."""
        if self._toggle_show_mip_button.text() == 'Show Max Intensity Proj':
            self._toggle_show_mip_button.setText('Hide Max Intensity Proj')
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
            self._toggle_show_mip_button.setText('Show Max Intensity Proj')
            for group in set(self._groups.values()):  # remove lines
                self._update_lines(group, only_2D=True)
        self._draw()

    def _toggle_show_max(self):
        """Toggle whether to color local maxima differently."""
        if self._toggle_show_max_button.text() == 'Show Maxima':
            self._toggle_show_max_button.setText('Hide Maxima')
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
            self._toggle_show_max_button.setText('Show Maxima')
        self._draw()

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
                    mri_data, cmap='hot', aspect='auto', alpha=0.25, zorder=2))
            self._toggle_brain_button.setText('Hide Brain')
        self._draw()

    def _key_press_event(self, event):
        """Execute functions when the user presses a key."""
        super(IntracranialElectrodeLocator, self)._key_press_event(event)

        if event.text() == 'm':
            self._mark_ch()

        if event.text() == 'r':
            self._remove_ch()

        if event.text() == 'b':
            self._toggle_show_brain()
