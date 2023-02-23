# -*- coding: utf-8 -*-
"""Source estimate viewing graphical user interfaces (GUIs)."""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

from qtpy import QtCore
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                            QMessageBox, QWidget, QSlider, QPushButton,
                            QComboBox, QLineEdit)
from matplotlib.colors import LinearSegmentedColormap

from ._core import SliceBrowser
from .. import BaseEpochs
from ..baseline import rescale, _check_baseline
from ..evoked import EvokedArray
from ..time_frequency import EpochsTFR
from ..io.constants import FIFF
from ..io.pick import _get_channel_types, _picks_to_idx, _pick_inst
from ..transforms import apply_trans
from ..utils import (_require_version, _validate_type, _check_range, fill_doc,
                     _check_option)
from ..viz.backends._utils import _qt_safe_window
from ..viz.utils import _get_cmap

BASE_INT_DTYPE = np.int16
COMPLEX_DTYPE = np.dtype([('re', BASE_INT_DTYPE),
                          ('im', BASE_INT_DTYPE)])
RANGE_VALUE = 2**15
RANGE_SQRT = 2**8  # round up so no overflow

VECTOR_SCALAR = 10
SLIDER_WIDTH = 300


def _check_consistent(items, name):
    if not len(items):
        return
    for item in items[1:]:
        if item != items[0]:
            raise RuntimeError(f'Inconsistent attribute {name}, '
                               f'got {items[0]} and {item}')
    return items[0]


def _get_src_lut(src):
    offset = 2 if src.kind == 'mixed' else 0
    inuse = [s['inuse'] for s in src[offset:]]
    rr = np.concatenate(
        [s['rr'][this_inuse] for s, this_inuse in zip(src[offset:], inuse)])
    shape = _check_consistent([this_src['shape'] for this_src in src],
                              "src['shape']")
    # order='F' so that F-order flattening is faster
    lut = -1 * np.ones(np.prod(shape), dtype=np.int64, order='F')
    n_vertices_seen = 0
    for this_inuse in inuse:
        this_inuse = this_inuse.astype(bool)
        n_vertices = np.sum(this_inuse)
        lut[this_inuse] = np.arange(
            n_vertices_seen, n_vertices_seen + n_vertices)
        n_vertices_seen += n_vertices
    lut = np.reshape(lut, shape, order='F')
    src_affine_ras = _check_consistent(
        [this_src['mri_ras_t']['trans'] for this_src in src],
        "src['mri_ras_t']")
    src_affine_src = _check_consistent(
        [this_src['src_mri_t']['trans'] for this_src in src],
        "src['src_mri_t']")
    affine = np.dot(src_affine_ras, src_affine_src)
    affine[:3] *= 1e3
    return lut, affine, src_affine_src * 1000, rr * 1000


def _make_vol(lut, stc_data):
    vol = np.zeros(lut.shape, dtype=stc_data.dtype, order='F') * np.nan
    vol[lut >= 0] = stc_data[lut[lut >= 0]]
    return vol.reshape(lut.shape, order='F')


def _coord_to_coord(coord, vox_ras_t, ras_vox_t):
    return apply_trans(ras_vox_t, apply_trans(vox_ras_t, coord))


def _threshold_array(array, min_val, max_val):
    array = array.astype(float)
    array[array < min_val] = np.nan
    array[array > max_val] = np.nan
    return array


def _int_complex_conj(data):
    # Since the mixed real * imaginary terms cancel out, the complex
    # conjugate is the same as squaring and adding the real and imaginary.
    # Pre-scale by the square root of the range so that the greatest
    # allowable value, when squared does not overflow.
    # Similarly, the divide by 2 allows for the greatest value in
    # both real and imaginary to be added without overflow
    return (data['re'] // RANGE_SQRT)**2 // 2 + \
           (data['im'] // RANGE_SQRT)**2 // 2


class VolSourceEstimateViewer(SliceBrowser):
    """View a source estimate time-course time-frequency visualization."""

    @_qt_safe_window(splash='_renderer.figure.splash', window='')
    def __init__(self, data, subject=None, subjects_dir=None, src=None,
                 inst=None, show_topomap=True, show=True, verbose=None):
        """View a volume time and/or frequency source time course estimate.

        Parameters
        ----------
        data : array-like
            An array of shape (``n_epochs``, ``n_sources``, ``n_ori``,
            ``n_freqs``, ``n_times``). ``n_epochs`` may be 1 for data
            averaged across epochs and ``n_freqs`` may be 1 for data
            that is in time only and is not time-frequency decomposed. For
            faster performance, data can be cast to integers or a
            custom complex data type that uses integers as done by
            :func:`mne.gui.view_vol_stc`.
        %(subject)s
        %(subjects_dir)s
        src : instance of SourceSpaces
            The volume source space for the ``stc``.
        inst : EpochsTFR | AverageTFR | None
            The time-frequency or data object to use to plot topography.
        show_topomap : bool
            Show the sensor topomap if ``True``.
        show : bool
            Show the GUI if ``True``.
        block : bool
            Whether to halt program execution until the figure is closed.
        %(verbose)s
        """
        _require_version('dipy', 'VolSourceEstimateViewer', '0.10.1')
        if src is None:
            raise NotImplementedError('`src` is required, surface source '
                                      'estimate viewing is not yet supported')
        if inst is None:
            raise NotImplementedError(
                '`data` as a source estimate object is '
                'not yet supported so `inst` is required')
        if not isinstance(data, np.ndarray) or data.ndim != 5:
            raise ValueError('`data` must be an array of dimensions '
                             '(n_epochs, n_sources, n_ori, n_freqs, n_times)')
        if isinstance(inst, (BaseEpochs, EpochsTFR)) and \
                data.shape[0] != len(inst):
            raise ValueError(
                'Number of epochs in `inst` does not match with `data`, '
                f'expected {data.shape[0]}, got {len(inst)}')
        n_src_verts = sum([this_src['nuse'] for this_src in src])
        if src is not None and data.shape[1] != n_src_verts:
            raise RuntimeError('Source vertices in `data` do not match with '
                               'source space vertices in `src`, '
                               f'expected {n_src_verts}, got {data.shape[1]}')
        if any([this_src['type'] == 'surf' for this_src in src]):
            raise NotImplementedError('Surface and mixed source space '
                                      'viewing is not implemented yet.')
        if not all([s['coord_frame'] == FIFF.FIFFV_COORD_MRI for s in src]):
            raise RuntimeError('The source space must be in the `mri`'
                               'coordinate frame')
        if hasattr(inst, 'freqs') and data.shape[3] != inst.freqs.size:
            raise ValueError(
                'Frequencies in `inst` do not match with `data`, '
                f'expected {data.shape[3]}, got {inst.freqs.size}')
        if hasattr(inst, 'freqs') and not (np.iscomplexobj(data) or
                                           data.dtype == COMPLEX_DTYPE):
            raise ValueError('Complex data is required for time-frequency '
                             'source estimates')
        if data.shape[4] != inst.times.size:
            raise ValueError(
                'Times in `inst` do not match with `data`, '
                f'expected {data.shape[4]}, got {inst.times.size}')
        self._verbose = verbose  # used for logging, unused here
        self._data = data
        self._src = src
        self._inst = inst
        self._show_topomap = show_topomap
        (self._src_lut, self._src_vox_scan_ras_t, self._src_vox_ras_t,
         self._src_rr) = _get_src_lut(src)
        self._src_scan_ras_vox_t = np.linalg.inv(self._src_vox_scan_ras_t)
        self._is_complex = np.iscomplexobj(self._data) or \
            self._data.dtype == COMPLEX_DTYPE
        self._baseline = 'none'
        self._bl_tmin = self._inst.times[0]
        self._bl_tmax = self._inst.times[-1]
        self._update = True  # can be set to False to prevent double updates
        # for time and frequency
        # check if only positive values will be used
        self._pos_support = self._is_complex or self._data.shape[2] > 1 or \
            (self._data >= 0).all()
        self._cmap = _get_cmap('hot' if self._pos_support else 'mne')

        # set default variables for plotting
        self._t_idx = self._inst.times.size // 2
        self._f_idx = self._inst.freqs.size // 2 \
            if hasattr(self._inst, 'freqs') else None
        self._alpha = 0.75
        self._epoch_idx = 'Average' + ' Power' * self._is_complex

        # initialize current 3D image for chosen time and frequency
        stc_data = self._pick_epoch(self._data)

        # take the vector magnitude, if scalar, does nothing
        self._stc_data_vol = np.linalg.norm(stc_data, axis=1)

        self._stc_min = np.nanmin(self._stc_data_vol)
        self._stc_range = np.nanmax(self._stc_data_vol) - self._stc_min

        stc_data_vol = self._pick_stc_tfr(self._stc_data_vol)
        self._stc_img = _make_vol(self._src_lut, stc_data_vol)

        super(VolSourceEstimateViewer, self).__init__(
            subject=subject, subjects_dir=subjects_dir)

        if src._subject != op.basename(self._subject_dir):
            raise RuntimeError(
                f'Source space subject ({src._subject})-freesurfer subject'
                f'({op.basename(self._subject_dir)}) mismatch')

        # make source time course plots
        self._images['stc'] = list()
        src_shape = np.array(self._src_lut.shape)
        corners = [  # center pixel on location
            _coord_to_coord(
                (0,) * 3, self._src_vox_scan_ras_t, self._scan_ras_vox_t),
            _coord_to_coord(
                src_shape - 1, self._src_vox_scan_ras_t, self._scan_ras_vox_t)
        ]
        src_coord = self._get_src_coord()
        for axis in range(3):
            stc_slice = np.take(self._stc_img, src_coord[axis], axis=axis).T
            x_idx, y_idx = self._xy_idx[axis]
            extent = [corners[0][x_idx], corners[1][x_idx],
                      corners[1][y_idx], corners[0][y_idx]]
            self._images['stc'].append(self._figs[axis].axes[0].imshow(
                stc_slice, aspect='auto', extent=extent, cmap=self._cmap,
                alpha=self._alpha, zorder=2))

        self._data_max = abs(stc_data).max()
        if self._data.shape[2] > 1 and not self._is_complex:
            # also compute vectors for chosen time
            self._stc_vectors = self._pick_stc_tfr(stc_data).astype(float)
            self._stc_vectors /= self._data_max
            self._stc_vectors_masked = self._stc_vectors.copy()

            assert self._data.shape[2] == 3
            self._vector_mapper, self._vector_data = self._renderer.quiver3d(
                *self._src_rr.T, *(VECTOR_SCALAR * self._stc_vectors_masked.T),
                color=None, mode='2darrow', scale_mode='vector', scale=1,
                opacity=1)
            self._vector_actor = self._renderer._actor(self._vector_mapper)
            self._vector_actor.GetProperty().SetLineWidth(2.)
            self._renderer.plotter.add_actor(self._vector_actor, render=False)

        # initialize 3D volumetric rendering
        # TO DO: add surface source space viewing as elif
        if any([this_src['type'] == 'vol' for this_src in self._src]):
            scalars = np.array(np.where(np.isnan(self._stc_img), 0, 1.))
            spacing = np.diag(self._src_vox_ras_t)[:3]
            origin = self._src_vox_ras_t[:3, 3] - spacing / 2.
            center = 0.5 * self._stc_range - self._stc_min
            self._grid, self._grid_mesh, self._volume_pos, self._volume_neg = \
                self._renderer._volume(
                    dimensions=src_shape, origin=origin,
                    spacing=spacing,
                    scalars=scalars.flatten(order='F'),
                    surface_alpha=self._alpha,
                    resolution=0.4, blending='mip', center=center)
            self._volume_pos_actor = self._renderer.plotter.add_actor(
                self._volume_pos, render=False)[0]
            self._volume_neg_actor = self._renderer.plotter.add_actor(
                self._volume_neg, render=False)[0]
            _, grid_prop = self._renderer.plotter.add_actor(
                self._grid_mesh, render=False)
            grid_prop.SetOpacity(0.1)
            self._scalar_bar = self._renderer.scalarbar(
                source=self._volume_pos_actor, n_labels=8, color='black',
                bgcolor='white', label_font_size=10)
            self._scalar_bar.SetOrientationToVertical()
            self._scalar_bar.SetHeight(0.6)
            self._scalar_bar.SetWidth(0.05)
            self._scalar_bar.SetPosition(0.02, 0.2)

        self._update_cmap()  # must be called for volume to render properly
        # keep focus on main window so that keypress events work
        self.setFocus()
        if show:
            self.show()

    def _get_min_max_val(self):
        """Get the minimum and maximum non-transparent values."""
        return [self._cmap_sliders[i].value() / SLIDER_WIDTH *
                self._stc_range + self._stc_min for i in (0, 2)]

    def _get_src_coord(self):
        """Get the current slice transformed to source space."""
        return tuple(np.round(_coord_to_coord(
            self._current_slice, self._vox_scan_ras_t,
            self._src_scan_ras_vox_t)).astype(int))

    def _update_stc_pick(self):
        """Update the normalized data with the epoch picked."""
        stc_data = self._pick_epoch(self._data)
        self._stc_data_vol = self._apply_vector_norm(stc_data)
        self._stc_data_vol = self._apply_baseline_correction(
            self._stc_data_vol)
        # deal with baseline infinite numbers
        inf_mask = np.isinf(self._stc_data_vol)
        if inf_mask.any():
            self._stc_data_vol[inf_mask] = np.nan
        self._stc_min = np.nanmin(self._stc_data_vol)
        self._stc_range = np.nanmax(self._stc_data_vol) - self._stc_min

    def _update_vectors(self):
        if self._data.shape[2] > 1 and not self._is_complex:
            # pick vector as well
            self._stc_vectors = self._pick_stc_tfr(self._data)
            self._stc_vectors = self._pick_epoch(
                self._stc_vectors).astype(float)
            self._stc_vectors /= self._data_max
            self._update_vector_threshold()
            self._plot_vectors()

    def _update_vector_threshold(self):
        """Update the threshold for the vectors."""
        # apply threshold, use same mask as for stc_img
        stc_data = self._pick_stc_tfr(self._stc_data_vol)
        min_val, max_val = self._get_min_max_val()
        self._stc_vectors_masked = self._stc_vectors.copy()
        self._stc_vectors_masked[stc_data < min_val] = np.nan
        self._stc_vectors_masked[stc_data > max_val] = np.nan

    def _update_stc_volume(self):
        """Select volume based on the current time, frequency and vertex."""
        stc_data = self._pick_stc_tfr(self._stc_data_vol)
        self._stc_img = _make_vol(self._src_lut, stc_data)
        self._stc_img = _threshold_array(
            self._stc_img, *self._get_min_max_val())

    def _update_stc_all(self):
        """Update the data in both the slice plots and the data plot."""
        # pick new epochs + baseline correction combination
        self._update_stc_pick()
        self._update_stc_images()  # and then make the new volume
        self._update_intensity()
        self._update_cmap()  # note: this updates stc slice plots
        self._plot_data()
        if self._show_topomap and self._update:
            self._plot_topomap()

    def _pick_stc_image(self):
        """Select time-(frequency) image based on vertex."""
        return self._pick_stc_vertex(self._stc_data_vol)

    def _pick_epoch(self, stc_data):
        """Select the source time course epoch based on the parameters."""
        if self._epoch_idx == 'Average':
            if stc_data.dtype == BASE_INT_DTYPE:
                stc_data = stc_data.mean(axis=0).astype(BASE_INT_DTYPE)
            else:
                stc_data = stc_data.mean(axis=0)
        elif self._epoch_idx == 'Average Power':
            if stc_data.dtype == COMPLEX_DTYPE:
                stc_data = np.sum(_int_complex_conj(
                    stc_data) // stc_data.shape[0], axis=0)
            else:
                stc_data = (stc_data * stc_data.conj()).real.mean(axis=0)
        elif self._epoch_idx == 'ITC':
            if stc_data.dtype == COMPLEX_DTYPE:
                stc_data = stc_data['re'].astype(np.complex64) + \
                    1j * stc_data['im']
                stc_data = np.abs((stc_data / np.abs(stc_data)).mean(axis=0))
            else:
                stc_data = np.abs((stc_data / np.abs(stc_data)).mean(axis=0))
        else:
            stc_data = stc_data[int(self._epoch_idx.replace('Epoch ', ''))]
            if stc_data.dtype == COMPLEX_DTYPE:
                stc_data = _int_complex_conj(stc_data)
            elif self._is_complex:
                stc_data = (stc_data * stc_data.conj()).real
        return stc_data

    def _apply_vector_norm(self, stc_data, axis=1):
        """Take the vector norm if source data is vector."""
        if self._epoch_idx == 'ITC':
            stc_data = np.max(stc_data, axis=axis)  # take maximum ITC
        elif stc_data.shape[axis] > 1:
            stc_data = np.linalg.norm(stc_data, axis=axis)  # take magnitude
            # if self._data.dtype in (COMPLEX_DTYPE, BASE_INT_DTYPE):
            #    stc_data = stc_data.round().astype(BASE_INT_DTYPE)
        else:
            stc_data = np.take(stc_data, 0, axis=axis)
        return stc_data

    def _apply_baseline_correction(self, stc_data):
        """Apply the chosen baseline correction to the data."""
        if self._baseline != 'none':  # do baseline correction
            stc_data = rescale(
                stc_data.astype(float), times=self._inst.times,
                baseline=(float(self._bl_tmin), float(self._bl_tmax)),
                mode=self._baseline, copy=True)
        return stc_data

    def _pick_stc_vertex(self, stc_data):
        """Select the vertex based on the cursor position."""
        src_coord = self._get_src_coord()
        if all([coord >= 0 and coord < dim for coord, dim in zip(
                src_coord, self._src_lut.shape)]) and \
                self._src_lut[src_coord] >= 0:
            stc_data = stc_data[self._src_lut[src_coord]]
        else:  # out-of-bounds or unused vertex
            stc_data = np.zeros(stc_data[:, 0].shape) * np.nan
        return stc_data

    def _pick_stc_tfr(self, stc_data):
        """Select the frequency and time based on GUI values."""
        stc_data = np.take(stc_data, self._t_idx, axis=-1)
        f_idx = 0 if self._f_idx is None else self._f_idx
        stc_data = np.take(stc_data, f_idx, axis=-1)
        return stc_data

    def _configure_ui(self):
        """Configure the main appearance of the user interface."""
        toolbar = self._configure_toolbar()
        slider_bar = self._configure_sliders()
        status_bar = self._configure_status_bar()
        data_plot = self._configure_data_plot()

        plot_vbox = QVBoxLayout()
        plot_vbox.addLayout(self._plt_grid)

        if self._show_topomap:
            data_hbox = QHBoxLayout()
            topo_plot = self._configure_topo_plot()
            data_hbox.addWidget(topo_plot)
            data_hbox.addWidget(data_plot)
            plot_vbox.addLayout(data_hbox)
        else:
            plot_vbox.addWidget(data_plot)

        main_hbox = QHBoxLayout()
        main_hbox.addLayout(slider_bar)
        main_hbox.addLayout(plot_vbox)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(toolbar)
        main_vbox.addLayout(main_hbox)
        main_vbox.addLayout(status_bar)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

    def _configure_toolbar(self):
        """Make a bar with buttons for user interactions."""
        hbox = QHBoxLayout()

        help_button = QPushButton('Help')
        help_button.released.connect(self._show_help)
        hbox.addWidget(help_button)

        hbox.addStretch(8)

        if self._data.shape[0] > 1:
            self._epoch_selector = QComboBox()
            if self._is_complex:
                self._epoch_selector.addItems(['Average Power'])
                self._epoch_selector.addItems(['ITC'])
            else:
                self._epoch_selector.addItems(['Average'])
            self._epoch_selector.addItems(
                [f'Epoch {i}' for i in range(self._data.shape[0])])
            self._epoch_selector.setCurrentText(self._epoch_idx)
            self._epoch_selector.currentTextChanged.connect(self._update_epoch)
            self._epoch_selector.setSizeAdjustPolicy(
                QComboBox.AdjustToContents)
            self._epoch_selector.keyPressEvent = self.keyPressEvent
            hbox.addWidget(self._epoch_selector)

        return hbox

    def _show_help(self):
        """Show the help menu."""
        QMessageBox.information(
            self, 'Help',
            "Help:\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "left angle bracket/right angle bracket: anterior/posterior")

    def _configure_sliders(self):
        """Make a bar with sliders on it."""

        def make_label(name):
            label = QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
            return label

        # modified from:
        # https://stackoverflow.com/questions/52689047/moving-qslider-to-mouse-click-position
        class Slider(QSlider):

            def mouseReleaseEvent(self, event):
                if event.button() == QtCore.Qt.LeftButton:
                    event.accept()
                    value = (self.maximum() - self.minimum()) * \
                        event.pos().x() / self.width() + self.minimum()
                    value = np.clip(value, 0, SLIDER_WIDTH)
                    self.setValue(int(round(value)))
                else:
                    super(Slider, self).mouseReleaseEvent(event)

        def make_slider(smin, smax, sval, sfun=None):
            slider = Slider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(round(smin)))
            slider.setMaximum(int(round(smax)))
            slider.setValue(int(round(sval)))
            slider.setTracking(False)  # only update on release
            if sfun is not None:
                slider.valueChanged.connect(sfun)
            slider.keyPressEvent = self.keyPressEvent
            slider.setMinimumWidth(SLIDER_WIDTH)
            return slider

        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(11, 11, 11, 11)  # for aesthetics

        if hasattr(self._inst, 'freqs'):
            slider_layout.addWidget(make_label('Frequency (Hz)'))
            self._freq_slider = make_slider(
                0, self._inst.freqs.size - 1, self._f_idx, self._update_freq)
            slider_layout.addWidget(self._freq_slider)
            freq_hbox = QHBoxLayout()
            freq_hbox.addWidget(make_label(str(self._inst.freqs[0].round(2))))
            freq_hbox.addStretch(1)
            freq_hbox.addWidget(make_label(str(self._inst.freqs[-1].round(2))))
            slider_layout.addLayout(freq_hbox)
            self._freq_label = make_label(
                f'Freq = {self._inst.freqs[self._f_idx].round(2)} Hz')
            slider_layout.addWidget(self._freq_label)
            slider_layout.addStretch(1)

        slider_layout.addWidget(make_label('Time (s)'))
        self._time_slider = make_slider(0, self._inst.times.size - 1,
                                        self._t_idx, self._update_time)
        slider_layout.addWidget(self._time_slider)
        time_hbox = QHBoxLayout()
        time_hbox.addWidget(make_label(str(self._inst.times[0].round(2))))
        time_hbox.addStretch(1)
        time_hbox.addWidget(make_label(str(self._inst.times[-1].round(2))))
        slider_layout.addLayout(time_hbox)
        self._time_label = make_label(
            f'Time = {self._inst.times[self._t_idx].round(2)} s')
        slider_layout.addWidget(self._time_label)
        slider_layout.addStretch(1)

        slider_layout.addWidget(make_label('Alpha'))
        self._alpha_slider = make_slider(
            0, SLIDER_WIDTH, int(self._alpha * SLIDER_WIDTH),
            self._update_alpha)
        slider_layout.addWidget(self._alpha_slider)
        self._alpha_label = make_label(f'Alpha = {self._alpha}')
        slider_layout.addWidget(self._alpha_label)
        slider_layout.addStretch(1)

        slider_layout.addWidget(make_label('min / mid / max'))
        self._cmap_sliders = [
            make_slider(0, SLIDER_WIDTH, 0, self._update_cmap),
            make_slider(0, SLIDER_WIDTH, SLIDER_WIDTH // 2,
                        self._update_cmap),
            make_slider(0, SLIDER_WIDTH, SLIDER_WIDTH, self._update_cmap)]
        for slider in self._cmap_sliders:
            slider_layout.addWidget(slider)
        slider_layout.addStretch(1)

        return slider_layout

    def _configure_status_bar(self, hbox=None):
        hbox = QHBoxLayout() if hbox is None else hbox

        bl_widget = QWidget()
        bl_widget.setStyleSheet('background-color: darkgray;')
        bl_hbox = QHBoxLayout()

        bl_hbox.addWidget(QLabel('Baseline'))
        self._baseline_selector = QComboBox()
        self._baseline_selector.addItems(['none', 'mean', 'ratio', 'logratio',
                                          'percent', 'zscore', 'zlogratio'])
        self._baseline_selector.setCurrentText('none')
        self._baseline_selector.currentTextChanged.connect(
            self._update_baseline)
        self._baseline_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._baseline_selector.keyPressEvent = self.keyPressEvent
        bl_hbox.addWidget(self._baseline_selector)

        bl_hbox.addWidget(QLabel('tmin ='))
        self._bl_tmin_textbox = QLineEdit(str(round(self._bl_tmin, 2)))
        self._bl_tmin_textbox.setMaximumWidth(40)
        self._bl_tmin_textbox.focusOutEvent = self._update_baseline_tmin
        bl_hbox.addWidget(self._bl_tmin_textbox)

        bl_hbox.addWidget(QLabel('tmax ='))
        self._bl_tmax_textbox = QLineEdit(str(round(self._bl_tmax, 2)))
        self._bl_tmax_textbox.setMaximumWidth(40)
        self._bl_tmax_textbox.focusOutEvent = self._update_baseline_tmax
        bl_hbox.addWidget(self._bl_tmax_textbox)

        bl_widget.setLayout(bl_hbox)

        hbox.addWidget(bl_widget)

        hbox.addStretch(3 if self._f_idx is None else 2)

        if self._show_topomap:
            hbox.addWidget(QLabel('Topo Data='))
            self._data_type_selector = QComboBox()
            self._data_type_selector.addItems(
                _get_channel_types(self._inst.info, picks='data', unique=True))
            self._data_type_selector.currentTextChanged.connect(
                self._update_data_type)
            self._data_type_selector.setSizeAdjustPolicy(
                QComboBox.AdjustToContents)
            self._data_type_selector.keyPressEvent = self.keyPressEvent
            hbox.addWidget(self._data_type_selector)
            hbox.addStretch(1)

        if self._f_idx is not None:
            hbox.addWidget(QLabel('Interpolate'))
            self._interp_button = QPushButton('On')
            self._interp_button.setMaximumWidth(25)  # not too big
            self._interp_button.setStyleSheet("background-color: green")
            hbox.addWidget(self._interp_button)
            self._interp_button.released.connect(self._toggle_interp)
            hbox.addStretch(1)

        self._go_to_extreme_button = QPushButton('Go to Max')
        self._go_to_extreme_button.released.connect(self.go_to_extreme)
        hbox.addWidget(self._go_to_extreme_button)
        hbox.addStretch(2)

        self._intensity_label = QLabel('')  # update later
        hbox.addWidget(self._intensity_label)

        # add SliceBrowser navigation items
        hbox = super(VolSourceEstimateViewer, self)._configure_status_bar(
            hbox=hbox)
        return hbox

    def _configure_data_plot(self):
        """Configure the plot that shows spectrograms/time-courses."""
        from ._core import _make_mpl_plot
        canvas, self._fig = _make_mpl_plot(
            dpi=96, tight=False, hide_axes=False, invert=False,
            facecolor='white')

        self._fig.axes[0].set_xlabel('Time (s)')
        self._fig.axes[0].set_xticks(
            [0, self._inst.times.size // 2, self._inst.times.size - 1])
        self._fig.axes[0].set_xticklabels(
            self._inst.times[[0, self._inst.times.size // 2, -1]].round(2))
        stc_data = self._pick_stc_image()
        if self._f_idx is None:
            self._fig.axes[0].set_facecolor('black')
            self._stc_plot = self._fig.axes[0].plot(
                stc_data[0], color='white')[0]
            self._stc_vline = self._fig.axes[0].axvline(
                x=self._t_idx, color='lime')
            self._fig.axes[0].set_ylabel('Activation (AU)')
            self._cax = None
        else:
            self._stc_plot = self._fig.axes[0].imshow(
                stc_data, aspect='auto', cmap=self._cmap,
                interpolation='bicubic')
            self._stc_vline = self._fig.axes[0].axvline(
                x=self._t_idx, color='lime', linewidth=0.5)
            self._stc_hline = self._fig.axes[0].axhline(
                y=self._f_idx, color='lime', linewidth=0.5)
            self._fig.axes[0].invert_yaxis()
            self._fig.axes[0].set_ylabel('Frequency (Hz)')
            self._fig.axes[0].set_yticks(range(self._inst.freqs.size))
            self._fig.axes[0].set_yticklabels(self._inst.freqs.round(2))
            self._cax = self._fig.add_axes([0.87, 0.25, 0.02, 0.7])
            self._cbar = self._fig.colorbar(self._stc_plot, cax=self._cax)
            self._cax.set_ylabel('Power')
        self._fig.axes[0].set_position([0.1, 0.25, 0.75, 0.7])
        self._fig.canvas.mpl_connect(
            'button_release_event', self._on_data_plot_click)
        canvas.setMinimumHeight(int(self.size().height() * 0.4))
        canvas.keyPressEvent = self.keyPressEvent
        return canvas

    def _plot_topomap(self):
        self._topo_fig.axes[0].clear()
        if isinstance(self._inst, EpochsTFR):
            inst_data = self._inst.data
        elif isinstance(self._inst, BaseEpochs):
            inst_data = self._inst.get_data()
        else:
            inst_data = self._inst.data[None]  # new axis for single epoch

        pick_idx = _picks_to_idx(
            self._inst.info, self._data_type_selector.currentText())
        inst_data = inst_data[:, pick_idx]

        evo_data = self._pick_epoch(inst_data)

        if self._f_idx is not None:
            evo_data = evo_data[:, self._f_idx]

        if self._baseline != 'none':
            evo_data = rescale(
                evo_data.astype(float), times=self._inst.times,
                baseline=(float(self._bl_tmin), float(self._bl_tmax)),
                mode=self._baseline, copy=False)

        info = _pick_inst(self._inst, self._data_type_selector.currentText(),
                          'bads').info
        ave = EvokedArray(evo_data, info, tmin=self._inst.times[0])

        ave.plot_topomap(times=self._inst.times[self._t_idx],
                         axes=self._topo_fig.axes[0], cmap=self._cmap,
                         colorbar=False, show=False)
        self._topo_fig.axes[0].set_title('')
        self._topo_fig.subplots_adjust(top=1.1, bottom=0.05)
        self._topo_fig.canvas.draw()

    def _configure_topo_plot(self):
        """Configure the plot that shows spectrograms/time-courses."""
        from ._core import _make_mpl_plot
        canvas, self._topo_fig = _make_mpl_plot(
            dpi=96, hide_axes=False, facecolor='white')
        # Topomap colorbar could be added later, a bit too much clutter though
        # self._topo_cax = self._topo_fig.add_axes((0.8, 0.1, 0.05, 0.75))
        self._plot_topomap()
        canvas.setMinimumHeight(int(self.size().height() * 0.4))
        canvas.setMaximumWidth(int(self.size().width() * 0.4))
        canvas.keyPressEvent = self.keyPressEvent
        return canvas

    def keyPressEvent(self, event):
        """Execute functions when the user presses a key."""
        super().keyPressEvent(event)

        # update if textbox done editing
        if event.key() == QtCore.Qt.Key_Return:
            for widget in (self._bl_tmin_textbox, self._bl_tmax_textbox):
                if widget.hasFocus():
                    widget.clearFocus()
                    self.setFocus()  # removing focus calls focus out event

    def _on_data_plot_click(self, event):
        """Update viewer when the data plot is clicked on."""
        if event.inaxes is self._fig.axes[0]:
            if self._f_idx is not None:
                self._update = False
                self.set_freq(self._inst.freqs[int(round(event.ydata))])
                self._update = True
            self.set_time(self._inst.times[int(round(event.xdata))])

    def set_baseline(self, baseline=None, mode=None):
        """Set the baseline.

        Parameters
        ----------
        baseline : array-like, shape (2,) | None
            The time interval to apply rescaling / baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
            Perform baseline correction by

            - subtracting the mean of baseline values ('mean')
            - dividing by the mean of baseline values ('ratio')
            - dividing by the mean of baseline values and taking the log
              ('logratio')
            - subtracting the mean of baseline values followed by dividing by
              the mean of baseline values ('percent')
            - subtracting the mean of baseline values and dividing by the
              standard deviation of baseline values ('zscore')
            - dividing by the mean of baseline values, taking the log, and
              dividing by the standard deviation of log baseline values
              ('zlogratio')
        tmin : float
            The minimum baseline time
        """  # noqa E501
        _check_option('mode', mode, ('mean', 'ratio', 'logratio', 'percent',
                                     'zscore', 'zlogratio', 'none', None))
        self._update = False
        self._baseline_selector.setCurrentText(
            'none' if mode is None else mode)
        if baseline is not None:
            baseline = _check_baseline(baseline, times=self._inst.times,
                                       sfreq=self._inst.info['sfreq'])
            tmin, tmax = baseline
            self._bl_tmin_textbox.setText(str(tmin))
            self._bl_tmax_textbox.setText(str(tmax))
        self._update = True
        self._update_stc_all()

    def _update_baseline(self, name):
        """Update the chosen baseline normalization method."""
        self._baseline = name
        self._cmap_sliders[0].setValue(0)
        self._cmap_sliders[1].setValue(SLIDER_WIDTH // 2)
        self._cmap_sliders[2].setValue(SLIDER_WIDTH)
        # all baselines have negative support
        self._cmap = _get_cmap('hot' if name == 'none' and self._pos_support
                               else 'mne')
        self._go_to_extreme_button.setText(
            'Go to Max' if name == 'none' and self._pos_support else
            'Go to Extreme')
        if self._update:  # don't update if bl_tmin, bl_tmax are also changing
            self._update_stc_all()

    def _update_baseline_tmin(self, event):
        """Update tmin for the baseline."""
        try:
            tmin = float(self._bl_tmin_textbox.text())
        except ValueError:
            self._bl_tmin_textbox.setText(str(round(self._bl_tmin, 2)))
        tmin = self._inst.times[np.clip(  # find nearest time
            self._inst.time_as_index(tmin, use_rounding=True)[0],
            0, self._inst.times.size - 1)]
        if tmin == self._bl_tmin:
            return
        self._bl_tmin = tmin
        if self._update:
            self._update_stc_all()

    def _update_baseline_tmax(self, event):
        """Update tmax for the baseline."""
        try:
            tmax = float(self._bl_tmax_textbox.text())
        except ValueError:
            self._bl_tmax_textbox.setText(str(round(self._bl_tmax, 2)))
            return
        tmax = self._inst.times[np.clip(  # find nearest time
            self._inst.time_as_index(tmax, use_rounding=True)[0],
            0, self._inst.times.size - 1)]
        if tmax == self._bl_tmax:
            return
        self._bl_tmax = tmax
        if self._update:
            self._update_stc_all()

    def _update_data_type(self, dtype):
        """Update which data type is shown in the topomap."""
        self._plot_topomap()

    def _update_data_plot_ylabel(self):
        """Update the ylabel of the data plot."""
        if self._epoch_idx == 'ITC':
            self._cax.set_ylabel('ITC')
        elif self._is_complex:
            self._cax.set_ylabel('Power')
        else:
            self._fig.axes[0].set_ylabel('Activation (AU)')

    def _update_epoch(self, name):
        """Change which epoch is viewed."""
        self._epoch_idx = name
        # handle plot labels
        self._update_data_plot_ylabel()
        # reset sliders
        if name == 'ITC' and self._epoch_idx != 'ITC':
            self._cmap_sliders[0].setValue(0)
            self._cmap_sliders[1].setValue(SLIDER_WIDTH // 2)
            self._cmap_sliders[2].setValue(SLIDER_WIDTH)
            self._baseline_selector.setCurrentText('none')

        if self._update:
            self._update_stc_all()
            self._update_vectors()

    def set_freq(self, freq):
        """Set the frequency to display (in Hz).

        Parameters
        ----------
        freq : float
            The frequency to show, in Hz.
        """
        if self._f_idx is None:
            raise ValueError('Source estimate does not contain frequencies')
        self._freq_slider.setValue(np.argmin(abs(self._inst.freqs - freq)))

    def _update_freq(self, event=None):
        """Update freq slider values."""
        self._f_idx = self._freq_slider.value()
        self._freq_label.setText(
            f'Freq = {self._inst.freqs[self._f_idx].round(2)} Hz')
        if self._update:
            self._update_stc_images()  # just need volume updated here
        self._stc_hline.set_ydata([self._f_idx])
        self._update_intensity()
        if self._show_topomap and self._update:
            self._plot_topomap()
        self._fig.canvas.draw()

    def set_time(self, time):
        """Set the time to display (in seconds).

        Parameters
        ----------
        time : float
            The time to show, in seconds.
        """
        self._time_slider.setValue(np.clip(
            self._inst.time_as_index(time, use_rounding=True)[0],
            0, self._inst.times.size - 1))

    def _update_time(self, event=None):
        """Update time slider values."""
        self._t_idx = self._time_slider.value()
        self._time_label.setText(
            f'Time = {self._inst.times[self._t_idx].round(2)} s')
        if self._update:
            self._update_stc_images()  # just need volume updated here
        self._stc_vline.set_xdata([self._t_idx])
        self._update_intensity()
        if self._show_topomap and self._update:
            self._plot_topomap()
        self._update_vectors()
        self._fig.canvas.draw()

    def set_alpha(self, alpha):
        """Set the opacity of the display.

        Parameters
        ----------
        alpha : float
            The opacity to use.
        """
        self._alpha_slider.setValue(np.clip(alpha, 0, 1))

    def _update_alpha(self, event=None):
        """Update stc plot alpha."""
        self._alpha = round(self._alpha_slider.value() / SLIDER_WIDTH, 2)
        self._alpha_label.setText(f'Alpha = {self._alpha}')
        for axis in range(3):
            self._images['stc'][axis].set_alpha(self._alpha)
        self._update_cmap()

    def set_cmap(self, vmin=None, vmid=None, vmax=None):
        """Update the colormap.

        Parameters
        ----------
        vmin : float
            The minimum color value relative to the selected data in [0, 1].
        vmin : float
            The middle color value relative to the selected data in [0, 1].
        vmin : float
            The maximum color value relative to the selected data in [0, 1].
        """
        for val, name in zip((vmin, vmid, vmax), ('vmin', 'vmid', 'vmax')):
            _validate_type(val, (int, float, None))

        self._update = False
        for i, val in enumerate((vmin, vmid, vmax)):
            if val is not None:
                _check_range(val, 0, 1, name)
                self._cmap_sliders[i].setValue(int(round(val * SLIDER_WIDTH)))
        self._update = True
        self._update_cmap()

    def _update_cmap(self, event=None, draw=True, update_slice_plots=True,
                     update_3d=True):
        """Update the colormap."""
        if not self._update:
            return

        # no recursive updating
        update_tmp = self._update
        self._update = False
        if self._cmap_sliders[0].value() > self._cmap_sliders[2].value():
            tmp = self._cmap_sliders[0].value()
            self._cmap_sliders[0].setValue(self._cmap_sliders[2].value())
            self._cmap_sliders[2].setValue(tmp)
        if self._cmap_sliders[1].value() > self._cmap_sliders[2].value():
            self._cmap_sliders[1].setValue(self._cmap_sliders[2].value())
        if self._cmap_sliders[1].value() < self._cmap_sliders[0].value():
            self._cmap_sliders[1].setValue(self._cmap_sliders[0].value())
        self._update = update_tmp

        vmin, vmid, vmax = [
            val / SLIDER_WIDTH * self._stc_range + self._stc_min
            for val in (self._cmap_sliders[i].value() for i in range(3))]
        mid_pt = (vmid - vmin) / (vmax - vmin)
        ctable = self._cmap(np.concatenate([
            np.linspace(0, mid_pt, 128), np.linspace(mid_pt, 1, 128)]))
        cmap = LinearSegmentedColormap.from_list('stc', ctable.tolist(), N=256)
        ctable = np.round(ctable * 255.0).astype(np.uint8)
        if self._stc_min < 0:  # make center values transparent
            zero_pt = np.argmin(abs(np.linspace(vmin, vmax, 256)))
            # 31 on either side of the zero point are made transparent
            ctable[max([zero_pt - 31, 0]):min([zero_pt + 32, 255]), 3] = 0
        else:  # make low values transparent
            ctable[:25, 3] = np.linspace(0, 255, 25)

        for axis in range(3):
            self._images['stc'][axis].set_clim(vmin, vmax)
            self._images['stc'][axis].set_cmap(cmap)
            if draw and self._update:
                self._figs[axis].canvas.draw()

        # update nans in slice plot image
        if update_slice_plots and self._update:
            self._update_stc_volume()
            self._plot_stc_images(draw=draw)

        if self._f_idx is None:
            self._fig.axes[0].set_ylim(
                [self._stc_min, self._stc_min + self._stc_range])
        else:
            self._stc_plot.set_clim(vmin, vmax)
            self._stc_plot.set_cmap(cmap)
            # update colorbar
            self._cax.clear()
            self._cbar = self._fig.colorbar(self._stc_plot, cax=self._cax)
            self._update_data_plot_ylabel()
        if draw and self._update:
            self._fig.canvas.draw()

        if not update_3d:
            return

        if self._data.shape[2] > 1 and not self._is_complex:
            # update vector mask
            self._update_vector_threshold()
            self._plot_vectors(draw=False)
            self._renderer._set_colormap_range(
                actor=self._vector_actor, ctable=ctable, scalar_bar=None,
                rng=[0, VECTOR_SCALAR])

        # set alpha
        ctable[ctable[:, 3] > self._alpha * 255, 3] = self._alpha * 255
        self._renderer._set_volume_range(self._volume_pos, ctable, self._alpha,
                                         self._scalar_bar, [vmin, vmax])
        self._renderer._set_volume_range(self._volume_neg, ctable, self._alpha,
                                         self._scalar_bar, [vmin, vmax])
        if draw and self._update:
            self._renderer._update()

    def go_to_extreme(self):
        """Go to the extreme intensity source vertex."""
        stc_idx, f_idx, t_idx = np.unravel_index(np.nanargmax(
            abs(self._stc_data_vol)), self._stc_data_vol.shape)
        if self._f_idx is not None:
            self._freq_slider.setValue(f_idx)
        self._time_slider.setValue(t_idx)
        max_coord = np.array(np.where(self._src_lut == stc_idx)).flatten()
        max_coord_mri = _coord_to_coord(
            max_coord, self._src_vox_scan_ras_t, self._scan_ras_vox_t)
        self._set_ras(apply_trans(self._vox_ras_t, max_coord_mri))

    def _plot_data(self, draw=True):
        """Update which coordinate's data is being shown."""
        stc_data = self._pick_stc_image()
        if self._f_idx is None:  # no freq data
            self._stc_plot.set_ydata(stc_data[0])
        else:
            self._stc_plot.set_data(stc_data)
        if draw and self._update:
            self._fig.canvas.draw()

    def _toggle_interp(self):
        """Toggle interpolating the spectrogram data plot."""
        if self._interp_button.text() == 'Off':
            self._interp_button.setText('On')
            self._interp_button.setStyleSheet("background-color: green")
        else:  # text == 'On', turn off
            self._interp_button.setText('Off')
            self._interp_button.setStyleSheet("background-color: red")

        self._stc_plot.set_interpolation(
            'bicubic' if self._interp_button.text() == 'On' else None)
        if self._update:
            self._fig.canvas.draw()
        # draws data plot, fixes vmin, vmax
        self._update_cmap(update_slice_plots=False, update_3d=False)

    def _update_intensity(self):
        """Update the intensity label."""
        label_str = '{:.3f}'
        if self._stc_range > 1e5:
            label_str = '{:.3e}'
        elif np.issubdtype(self._stc_img.dtype, np.integer):
            label_str = '{:d}'
        self._intensity_label.setText(
            ('intensity = ' + label_str).format(
                self._stc_img[tuple(self._get_src_coord())]))

    def _update_moved(self):
        """Update when cursor position changes."""
        super()._update_moved()
        self._update_intensity()

    @fill_doc
    def set_3d_view(self, roll=None, distance=None, azimuth=None,
                    elevation=None, focalpoint=None):
        """Orient camera to display view.

        Parameters
        ----------
        %(roll)s
        %(distance)s
        %(azimuth)s
        %(elevation)s
        %(focalpoint)s
        """
        self._renderer.set_camera(
            roll=roll, distance=distance, azimuth=azimuth,
            elevation=elevation, focalpoint=focalpoint, reset_camera=False)
        self._renderer._update()

    def _plot_vectors(self, draw=True):
        """Update the vector plots."""
        if self._data.shape[2] > 1 and not self._is_complex:
            self._vector_data.point_data['vec'] = \
                VECTOR_SCALAR * self._stc_vectors_masked
            if draw and self._update:
                self._renderer._update()

    def _update_stc_images(self, draw=True):
        """Update the stc image based on the time and frequency range."""
        self._update_stc_volume()
        self._plot_stc_images(draw=draw)
        self._plot_3d_stc(draw=draw)

    def _plot_3d_stc(self, draw=True):
        """Update the 3D rendering."""
        self._plot_vectors(draw=False)
        self._grid.cell_data['values'] = np.where(
            np.isnan(self._stc_img), 0., self._stc_img).flatten(order='F')
        if draw and self._update:
            self._renderer._update()

    def _plot_stc_images(self, axis=None, draw=True):
        """Update the stc image(s)."""
        src_coord = self._get_src_coord()
        for axis in range(3):
            # ensure in bounds
            if src_coord[axis] >= 0 and \
                    src_coord[axis] < self._stc_img.shape[axis]:
                stc_slice = np.take(
                    self._stc_img, src_coord[axis], axis=axis).T
            else:
                stc_slice = np.take(self._stc_img, 0, axis=axis).T * np.nan
            self._images['stc'][axis].set_data(stc_slice)
            if draw and self._update:
                self._draw(axis)

    def _update_images(self, axis=None, draw=True):
        """Update images when general changes happen."""
        self._plot_stc_images(axis=axis, draw=draw)
        self._plot_data(draw=draw)
        super()._update_images()
