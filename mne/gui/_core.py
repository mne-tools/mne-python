# -*- coding: utf-8 -*-
"""Shared GUI classes and functions."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
from functools import partial

from qtpy import QtCore
from qtpy.QtCore import Slot
from qtpy.QtWidgets import (QMainWindow, QGridLayout,
                            QVBoxLayout, QHBoxLayout, QLabel,
                            QMessageBox, QWidget, QPlainTextEdit)

from matplotlib import patheffects
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .._freesurfer import _import_nibabel
from ..viz.backends.renderer import _get_renderer
from ..viz.utils import safe_event
from ..surface import _read_mri_surface, _marching_cubes
from ..transforms import apply_trans, _frame_to_str
from ..utils import logger, _check_fname, verbose, warn, get_subjects_dir

_IMG_LABELS = [['I', 'P'], ['I', 'L'], ['P', 'L']]
_ZOOM_STEP_SIZE = 5


@verbose
def _load_image(img, verbose=None):
    """Load data from a 3D image file (e.g. CT, MR)."""
    nib = _import_nibabel('use GUI')
    if not isinstance(img, nib.spatialimages.SpatialImage):
        logger.info(f'Loading {img}')
        _check_fname(img, overwrite='read', must_exist=True)
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


class SliceBrowser(QMainWindow):
    """Navigate between slices of an MRI, CT, etc. image."""

    _xy_idx = (
        (1, 2),
        (0, 2),
        (0, 1),
    )

    def __init__(self, base_image=None, subject=None, subjects_dir=None,
                 verbose=None):
        """GUI for browsing slices of anatomical images."""
        # initialize QMainWindow class
        super(SliceBrowser, self).__init__()

        self._verbose = verbose
        # if bad/None subject, will raise an informative error when loading MRI
        subject = os.environ.get('SUBJECT') if subject is None else subject
        subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
        self._subject_dir = op.join(subjects_dir, subject)
        self._load_image_data(base_image=base_image)

        # GUI design

        # Main plots: make one plot for each view; sagittal, coronal, axial
        self._plt_grid = QGridLayout()
        self._figs = list()
        for i in range(3):
            canvas, fig = _make_slice_plot()
            self._plt_grid.addWidget(canvas, i // 2, i % 2)
            self._figs.append(fig)
        self._renderer = _get_renderer(
            name='Slice Browser', size=(400, 400), bgcolor='w')
        self._plt_grid.addWidget(self._renderer.plotter, 1, 1)

        self._set_ras([0., 0., 0.], update_plots=False)

        self._plot_images()

        self._configure_ui()

    def _configure_ui(self):
        bottom_hbox = self._configure_status_bar()

        # Put everything together
        plot_ch_hbox = QHBoxLayout()
        plot_ch_hbox.addLayout(self._plt_grid)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(plot_ch_hbox)
        main_vbox.addLayout(bottom_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

    def _load_image_data(self, base_image=None):
        """Get image data to display and transforms to/from vox/RAS."""
        # allows recon-all not to be finished (T1 made in a few minutes)
        mri_img = 'brain' if op.isfile(op.join(
            self._subject_dir, 'mri', 'brain.mgz')) else 'T1'
        self._mri_data, self._vox_ras_t = _load_image(
            op.join(self._subject_dir, 'mri', f'{mri_img}.mgz'))
        self._ras_vox_t = np.linalg.inv(self._vox_ras_t)

        self._voxel_sizes = np.array(self._mri_data.shape)

        # We need our extents to land the centers of each pixel on the voxel
        # number. This code assumes 1mm isotropic...
        img_delta = 0.5
        self._img_extents = list(
            [-img_delta, self._voxel_sizes[idx[0]] - img_delta,
             -img_delta, self._voxel_sizes[idx[1]] - img_delta]
            for idx in self._xy_idx)

        # ready alternate base image if provided, otherwise use brain/T1
        if base_image is None:
            self._base_data = self._mri_data
        else:
            self._base_data, vox_ras_t = _load_image(base_image)
            if self._mri_data.shape != self._base_data.shape or \
                    not np.allclose(self._vox_ras_t, vox_ras_t, rtol=1e-6):
                raise ValueError('Base image is not aligned to MRI, got '
                                 f'Base shape={self._base_data.shape}, '
                                 f'MRI shape={self._mri_data.shape}, '
                                 f'Base affine={vox_ras_t} and '
                                 f'MRI affine={self._vox_ras_t}')

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

    def _plot_images(self):
        """Use the MRI or CT to make plots."""
        # Plot sagittal (0), coronal (1) or axial (2) view
        self._images = dict(base=list(), cursor_v=list(), cursor_h=list(),
                            bounds=list())
        img_min = np.nanmin(self._base_data)
        img_max = np.nanmax(self._base_data)
        text_kwargs = dict(fontsize='medium', weight='bold', color='#66CCEE',
                           family='monospace', ha='center', va='center',
                           path_effects=[patheffects.withStroke(
                               linewidth=4, foreground="k", alpha=0.75)])
        xyz = apply_trans(self._ras_vox_t, self._ras)
        for axis in range(3):
            plot_x_idx, plot_y_idx = self._xy_idx[axis]
            fig = self._figs[axis]
            ax = fig.axes[0]
            img_data = np.take(self._base_data, self._current_slice[axis],
                               axis=axis).T
            self._images['base'].append(ax.imshow(
                img_data, cmap='gray', aspect='auto', zorder=1,
                vmin=img_min, vmax=img_max))
            img_extent = self._img_extents[axis]  # x0, x1, y0, y1
            w, h = np.diff(np.array(img_extent).reshape(2, 2), axis=1)[:, 0]
            self._images['bounds'].append(Rectangle(
                img_extent[::2], w, h, edgecolor='w', facecolor='none',
                alpha=0.25, lw=0.5, zorder=1.5))
            ax.add_patch(self._images['bounds'][-1])
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
                self._base_data < np.quantile(self._base_data, 0.95), 0, 1),
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
        self._renderer.set_camera(azimuth=90, elevation=90, distance=300,
                                  focalpoint=tuple(self._ras))
        # update plots
        self._draw()
        self._renderer._update()

    def _configure_status_bar(self, hbox=None):
        """Make a bar at the bottom with information in it."""
        hbox = QHBoxLayout() if hbox is None else hbox

        self._intensity_label = QLabel('')  # update later
        hbox.addWidget(self._intensity_label)

        VOX_label = QLabel('VOX =')
        self._VOX_textbox = QPlainTextEdit('')  # update later
        self._VOX_textbox.setMaximumHeight(25)
        self._VOX_textbox.setMaximumWidth(125)
        self._VOX_textbox.focusOutEvent = self._update_VOX
        self._VOX_textbox.textChanged.connect(self._check_update_VOX)
        hbox.addWidget(VOX_label)
        hbox.addWidget(self._VOX_textbox)

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

    def _update_camera(self, render=False):
        """Update the camera position."""
        self._renderer.set_camera(
            # needs fix, distance moves when focal point updates
            distance=self._renderer.plotter.camera.distance * 0.9,
            focalpoint=tuple(self._ras),
            reset_camera=False)

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

    @Slot()
    def _update_RAS(self, event):
        """Interpret user input to the RAS textbox."""
        text = self._RAS_textbox.toPlainText()
        ras = self._convert_text(text, 'ras')
        if ras is not None:
            self._set_ras(ras)

    @Slot()
    def _update_VOX(self, event):
        """Interpret user input to the RAS textbox."""
        text = self._VOX_textbox.toPlainText()
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

    @Slot()
    def _check_update_RAS(self):
        """Check whether the RAS textbox is done being edited."""
        if '\n' in self._RAS_textbox.toPlainText():
            self._update_RAS(event=None)

    @Slot()
    def _check_update_VOX(self):
        """Check whether the VOX textbox is done being edited."""
        if '\n' in self._VOX_textbox.toPlainText():
            self._update_VOX(event=None)

    def _draw(self, axis=None):
        """Update the figures with a draw call."""
        for axis in (range(3) if axis is None else [axis]):
            self._figs[axis].canvas.draw()

    def _update_base_images(self, axis=None, draw=False):
        """Update the base images."""
        for axis in range(3) if axis is None else [axis]:
            img_data = np.take(self._base_data, self._current_slice[axis],
                               axis=axis).T
            self._images['base'][axis].set_data(img_data)
            if draw:
                self._draw(axis)

    def _update_images(self, axis=None, draw=True):
        """Update CT and channel images when general changes happen."""
        self._update_base_images(axis=axis)
        if draw:
            self._draw(axis)

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
        QMessageBox.information(
            self, 'Help',
            "Help:\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "left angle bracket/right angle bracket: anterior/posterior")

    def _key_press_event(self, event):
        """Execute functions when the user presses a key."""
        if event.key() == 'escape':
            self.close()

        if event.text() == 'h':
            self._show_help()

        if event.text() in ('=', '+', '-'):
            self._zoom(sign=-2 * (event.text() == '-') + 1, draw=True)

        # Changing slices
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                           QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                           QtCore.Qt.Key_Comma, QtCore.Qt.Key_Period,
                           QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
            ras = np.array(self._ras)
            if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                ras[2] += 2 * (event.key() == QtCore.Qt.Key_Up) - 1
            elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                ras[0] += 2 * (event.key() == QtCore.Qt.Key_Right) - 1
            else:
                ras[1] += 2 * (event.key() == QtCore.Qt.Key_PageUp or
                               event.key() == QtCore.Qt.Key_Period) - 1
            self._set_ras(ras)

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
        self._RAS_textbox.setPlainText('{:.2f}, {:.2f}, {:.2f}'.format(
            *self._ras))
        self._VOX_textbox.setPlainText('{:3d}, {:3d}, {:3d}'.format(
            *self._current_slice))
        self._intensity_label.setText('intensity = {:.2f}'.format(
            self._base_data[tuple(self._current_slice)]))

    @safe_event
    def closeEvent(self, event):
        """Clean up upon closing the window."""
        self._renderer.plotter.close()
        self.close()
