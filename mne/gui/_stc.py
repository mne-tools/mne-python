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
                            QComboBox)

from ._core import SliceBrowser
from ..io.constants import FIFF
from ..transforms import apply_trans
from ..utils import _require_version, _validate_type, _check_range, fill_doc
from ..viz.utils import _get_cmap

COMPLEX_DTYPE = np.dtype([('re', np.int64), ('im', np.int64)])


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


def _min_range(data):
    this_min = np.nanmin(data)
    this_range = np.nanmax(data) - this_min
    return this_range, this_min


class VolSourceEstimateViewer(SliceBrowser):
    """View a source estimate time-course time-frequency visualization."""

    def __init__(self, data, subject=None, subjects_dir=None, src=None,
                 inst=None, show=True, verbose=None):
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
        if 'Epochs' in str(type(inst)) and data.shape[0] > 1 and \
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
        if data.shape[4] != inst.times.size:
            raise ValueError(
                'Times in `inst` do not match with `data`, '
                f'expected {data.shape[4]}, got {inst.times.size}')
        self._verbose = verbose  # used for logging, unused here
        self._data = data
        self._src = src
        self._inst = inst
        (self._src_lut, self._src_vox_scan_ras_t, self._src_vox_ras_t,
         self._src_rr) = _get_src_lut(src)
        self._src_scan_ras_vox_t = np.linalg.inv(self._src_vox_scan_ras_t)
        self._is_complex = np.iscomplexobj(self._data) or \
            self._data.dtype == COMPLEX_DTYPE
        # check if only positive values will be used
        pos_support = self._is_complex or self._data.shape[2] > 1 or \
            (self._data >= 0).all()
        self._cmap = _get_cmap('hot' if pos_support else 'mne')

        # set default variables for plotting
        self._t_idx = self._inst.times.size // 2
        self._f_idx = self._inst.freqs.size // 2 \
            if hasattr(self._inst, 'freqs') else None
        self._alpha = 0.75
        self._epoch_idx = 'Average' + ' Power' * self._is_complex

        # initialize current 3D image for chosen time and frequency
        stc_data = self._pick_stc_epoch()
        stc_data_vol = np.linalg.norm(stc_data, axis=1)

        self._stc_min = np.nanmin(stc_data_vol)
        self._stc_range = np.nanmax(stc_data_vol) - self._stc_min

        # take the vector magnitude, if scalar, does nothing
        stc_data_vol = self._pick_stc_tfr(stc_data_vol)
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
                stc_slice, cmap=self._cmap, aspect='auto', extent=extent,
                alpha=self._alpha, zorder=2))

        # plot vectors if vector stc
        if self._data.shape[2] > 1:
            assert self._data.shape[2] == 3
            vectors = self._pick_stc_tfr(stc_data)
            # rescale
            vectors = 5 * vectors / (self._stc_min + self._stc_range)
            self._vector_mapper, self._vector_data = self._renderer.quiver3d(
                *self._src_rr.T, *vectors.T, color=None, mode='2darrow',
                scale_mode='vector', scale=1, opacity=1)
            self._vector_actor = self._renderer._actor(self._vector_mapper)
            self._vector_actor.GetProperty().SetLineWidth(2.)
            self._renderer.plotter.add_actor(self._vector_actor, render=False)

        # initialize 3D volumetric rendering
        # TO DO: add surface source space viewing as elif
        if any([this_src['type'] == 'vol' for this_src in self._src]):
            scalars = np.array(np.where(np.isnan(self._stc_img), 0., 1.))
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
            self._volume_neg_actor = None if self._volume_neg is not None \
                else self._renderer.plotter.add_actor(
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

        if show:
            self.show()

    def _get_src_coord(self):
        """Get the current slice transformed to source space."""
        return tuple(np.round(_coord_to_coord(
            self._current_slice, self._vox_scan_ras_t,
            self._src_scan_ras_vox_t)).astype(int))

    def _pick_stc_epoch(self):
        """Select the source time course epoch based on the parameters."""
        if self._epoch_idx == 'Average':
            stc_data = self._data.mean(axis=0)
        elif self._epoch_idx == 'Average Power':
            if self._data.dtype == COMPLEX_DTYPE:
                stc_data = (stc_data['re'] + 1j * stc_data['im']) * \
                    (stc_data['re'] - 1j * stc_data['im']).astype(
                        np.int64).mean(axis=0)
            else:
                stc_data = (self._data * self._data.conj()).real.mean(axis=0)
        elif self._epoch_idx == 'ITC':
            if self._data.dtype == COMPLEX_DTYPE:
                stc_data = np.abs(
                    (stc_data['re'] + 1j * stc_data['im']) /
                    np.abs(stc_data['re'] - 1j * stc_data['im'])).astype(
                        np.int64).mean(axis=0)
            else:
                stc_data = np.abs((self._data / np.abs(self._data)).mean(
                    axis=0))
        else:
            stc_data = self._data[int(self._epoch_idx.replace('Epoch ', ''))]
            if self._is_complex:
                if stc_data.dtype == COMPLEX_DTYPE:
                    stc_data = (stc_data['re'] + 1j * stc_data['im']) * \
                        (stc_data['re'] - 1j * stc_data['im']).astype(np.int64)
                else:
                    stc_data = (stc_data * stc_data.conj()).real
        return stc_data

    def _pick_stc_vertex(self, stc_data):
        """Select the vertex based on the cursor position."""
        src_coord = self._get_src_coord()
        if all([coord >= 0 and coord < dim for coord, dim in zip(
                src_coord, self._src_lut.shape)]) and \
                self._src_lut[src_coord] >= 0:
            stc_data = stc_data[self._src_lut[src_coord]]
        else:  # out-of-bounds or unused vertex
            stc_data = stc_data[0] * np.nan  # pick the first one, make nan
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
            self._epoch_selector.addItems(
                [f'Epoch {i}' for i in range(self._data.shape[0])])
            if np.iscomplexobj(self._data):
                self._epoch_selector.addItems(['Average Power'])
                if self._data.shape[2] == 1:  # only allow ITC for scalar
                    self._epoch_selector.addItems(['ITC'])
            else:
                self._epoch_selector.addItems(['Average'])
            self._epoch_selector.setCurrentText(self._epoch_idx)
            self._epoch_selector.currentTextChanged.connect(self._update_epoch)
            self._epoch_selector.setSizeAdjustPolicy(
                QComboBox.AdjustToContents)
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

        slider_layout = QVBoxLayout()

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
        self._alpha_slider = make_slider(0, 100, int(self._alpha * 100),
                                         self._update_alpha)
        slider_layout.addWidget(self._alpha_slider)
        self._alpha_label = make_label(f'Alpha = {self._alpha}')
        slider_layout.addWidget(self._alpha_label)
        slider_layout.addStretch(1)

        slider_layout.addWidget(make_label('min / mid / max'))
        self._cmap_sliders = [
            make_slider(0, 1000, 0, self._update_cmap),
            make_slider(0, 1000, 500, self._update_cmap),
            make_slider(0, 1000, 1000, self._update_cmap)]
        for slider in self._cmap_sliders:
            slider_layout.addWidget(slider)
        slider_layout.addStretch(1)

        return slider_layout

    def _configure_status_bar(self, hbox=None):
        hbox = QHBoxLayout() if hbox is None else hbox

        hbox.addStretch(3)

        self._go_to_max_button = QPushButton('Go to Maxima')
        self._go_to_max_button.released.connect(self.go_to_max)
        hbox.addWidget(self._go_to_max_button)

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
            dpi=96, tight=False, hide_axes=False)
        stc_data = self._pick_stc_epoch()
        stc_data = np.linalg.norm(stc_data, axis=1)  # take magnitude
        stc_data = self._pick_stc_vertex(stc_data)

        self._fig.axes[0].set_position([0.1, 0.25, 0.75, 0.6])
        self._fig.axes[0].set_xlabel('Time (s)')
        self._fig.axes[0].set_xticks(
            [0, self._inst.times.size // 2, self._inst.times.size - 1])
        self._fig.axes[0].set_xticklabels(
            self._inst.times[[0, self._inst.times.size // 2, -1]].round(2))
        if self._f_idx is None:
            self._stc_plot = self._fig.axes[0].plot(stc_data[0])[0]
            self._stc_vline = self._fig.axes[0].axvline(
                x=self._t_idx, color='yellow')
            self._fig.axes[0].set_ylabel('Activation (AU)')
            self._cax = None
        else:
            self._stc_plot = self._fig.axes[0].imshow(
                stc_data, aspect='auto', cmap=self._cmap,
                interpolation='bicubic')
            self._stc_vline = self._fig.axes[0].axvline(
                x=self._t_idx, color='white', linewidth=0.5)
            self._stc_hline = self._fig.axes[0].axhline(
                y=self._f_idx, color='white', linewidth=0.5)
            self._cax = self._fig.colorbar(
                self._stc_plot, ax=self._fig.axes[0])
            self._cax.ax.set_ylabel('Power')
            self._fig.axes[0].invert_yaxis()
            self._fig.axes[0].set_ylabel('Frequency (Hz)')
            self._fig.axes[0].set_yticks(range(self._inst.freqs.size))
            self._fig.axes[0].set_yticklabels(self._inst.freqs.round(2))
        self._fig.canvas.mpl_connect(
            'button_release_event', self._on_data_plot_click)
        canvas.setMinimumHeight(int(self.size().height() * 0.4))
        return canvas

    def _on_data_plot_click(self, event):
        """Update viewer when the data plot is clicked on."""
        if event.inaxes is self._fig.axes[0]:
            self.set_time(self._inst.times[int(round(event.xdata))])
            if self._f_idx is not None:
                self.set_freq(self._inst.freqs[int(round(event.ydata))])

    def _update_epoch(self, name):
        """Change which epoch is viewed."""
        self._epoch_idx = name
        # handle plot labels
        if self._epoch_idx == 'ITC':
            self._cax.ax.set_ylabel('ITC')
        elif self._is_complex:
            self._cax.ax.set_ylabel('Power')
        else:
            self._fig.axes[0].set_ylabel('Activation (AU)')
        # reset sliders
        if name == 'ITC' != self._epoch_idx == 'ITC':
            self._cmap_sliders[0].setValue(0)
            self._cmap_sliders[1].setValue(500)
            self._cmap_sliders[2].setValue(1000)
        self._update_stc_image()

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

    def _update_freq(self):
        """Update freq slider values."""
        self._f_idx = self._freq_slider.value()
        self._freq_label.setText(
            f'Freq = {self._inst.freqs[self._f_idx].round(2)} Hz')
        self._update_stc_image()
        self._stc_hline.set_ydata(self._f_idx)
        self._fig.canvas.draw()

    def set_time(self, time):
        """Set the time to display (in seconds).

        Parameters
        ----------
        time : float
            The time to show, in seconds.
        """
        self._time_slider.setValue(self._inst.time_as_index(time)[0])

    def _update_time(self):
        """Update time slider values."""
        self._t_idx = self._time_slider.value()
        self._time_label.setText(
            f'Time = {self._inst.times[self._t_idx].round(2)} s')
        self._update_stc_image()
        self._stc_vline.set_xdata(self._t_idx)
        self._fig.canvas.draw()

    def set_alpha(self, alpha):
        """Set the opacity of the display.

        Parameters
        ----------
        alpha : float
            The opacity to use.
        """
        self._alpha_slider.setValue(np.clip(alpha, 0, 1))

    def _update_alpha(self):
        """Update stc plot alpha."""
        self._alpha = self._alpha_slider.value() / 100
        self._alpha_label.setText(f'Alpha = {self._alpha}')
        for axis in range(3):
            self._images['stc'][axis].set_alpha(self._alpha)
        self._update_cmap()

    def update_cmap(self, vmin=None, vmid=None, vmax=None):
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

        for i, val in enumerate((vmin, vmid, vmax)):
            if val is not None:
                _check_range(val, 0, 1, name)
                self._cmap_sliders[i].setValue(int(round(val * 1000)))

    def _update_cmap(self):
        """Update colormap."""
        if self._cmap_sliders[0].value() > self._cmap_sliders[2].value():
            tmp = self._cmap_sliders[0].value()
            self._cmap_sliders[0].setValue(self._cmap_sliders[2].value())
            self._cmap_sliders[2].setValue(tmp)
        if self._cmap_sliders[1].value() > self._cmap_sliders[2].value():
            self._cmap_sliders[1].setValue(self._cmap_sliders[2].value())
        if self._cmap_sliders[1].value() < self._cmap_sliders[0].value():
            self._cmap_sliders[1].setValue(self._cmap_sliders[0].value())
        vmin, vmax = [val / 1000 * self._stc_range + self._stc_min
                      for val in (self._cmap_sliders[0].value(),
                                  self._cmap_sliders[2].value())]
        for axis in range(3):
            self._images['stc'][axis].set_clim(vmin, vmax)
            self._figs[axis].canvas.draw()
        if self._f_idx is None:
            self._fig.axes[0].set_ylim([vmin, vmax])
        else:
            self._stc_plot.set_clim(vmin, vmax)
        self._fig.canvas.draw()

        ctable = np.round(self._cmap(
            np.linspace(0, 1, 256)) * 255.0).astype(np.uint8)
        if self._stc_min < 0:  # make center values transparent
            ctable[97:159, 3] = 0  # 31 on either side of center (128)
        else:  # make low values transparent
            ctable[:25, 3] = np.linspace(0, 255, 25)

        if self._data.shape[2] > 1:
            self._renderer._set_colormap_range(
                actor=self._vector_actor, ctable=ctable, scalar_bar=None,
                rng=[vmin, vmax])

        # set alpha
        ctable[ctable[:, 3] > self._alpha * 255, 3] = self._alpha * 255
        self._renderer._set_volume_range(self._volume_pos, ctable, self._alpha,
                                         self._scalar_bar, [vmin, vmax])
        self._renderer._update()

    def go_to_max(self):
        """Go to the maximum intensity source vertex."""
        stc_data = self._pick_stc_epoch()
        stc_data = np.linalg.norm(stc_data, axis=1)  # vector magnitude
        stc_idx, f_idx, t_idx = \
            np.unravel_index(np.nanargmax(stc_data), stc_data.shape)
        if self._f_idx is not None:
            self._freq_slider.setValue(f_idx)
        self._time_slider.setValue(t_idx)
        max_coord = np.array(np.where(self._src_lut == stc_idx)).flatten()
        max_coord_mri = _coord_to_coord(
            max_coord, self._src_vox_scan_ras_t, self._scan_ras_vox_t)
        self._set_ras(apply_trans(self._vox_ras_t, max_coord_mri))

    def _update_data_plot(self, draw=False):
        """Update which coordinate's data is being shown."""
        stc_data = self._pick_stc_epoch()
        stc_data = np.linalg.norm(stc_data, axis=1)  # vector magnitude
        stc_data = self._pick_stc_vertex(stc_data)
        if self._f_idx is None:  # no freq data
            self._stc_plot.set_ydata(stc_data[0])
        else:
            self._stc_plot.set_data(stc_data)
        if draw:
            self._fig.canvas.draw()

    def _update_moved(self):
        """Update when cursor position changes."""
        super()._update_moved()
        self._intensity_label.setText(
            'intensity = ' +
            ('{:.3e}' if self._stc_range > 1e5 else '{:.3f}').format(
                self._stc_img[tuple(self._get_src_coord())]))

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

    def _update_stc_image(self):
        """Update the stc image based on the time and frequency range."""
        stc_data = self._pick_stc_epoch()
        stc_data_vol = np.linalg.norm(stc_data, axis=1)

        self._stc_min = np.nanmin(stc_data_vol)
        self._stc_range = np.nanmax(stc_data_vol) - self._stc_min

        stc_data_vol = self._pick_stc_tfr(stc_data_vol)
        self._stc_img = _make_vol(self._src_lut, stc_data_vol)

        if self._data.shape[2] > 1:
            vectors = self._pick_stc_tfr(stc_data)
            # rescale
            vectors = 5 * vectors / (self._stc_min + self._stc_range)
            self._vector_data.point_data['vec'] = vectors

        self._grid.cell_data['values'] = np.where(
            np.isnan(self._stc_img), 0., self._stc_img).flatten(order='F')
        self._update_images()
        self._update_cmap()

    def _update_stc_images(self, axis=None, draw=False):
        """Update the stc image(s)."""
        src_coord = self._get_src_coord()
        for axis in range(3):
            # ensure in bounds
            if src_coord[axis] >= 0 and \
                    src_coord[axis] < self._stc_img.shape[axis]:
                stc_data = np.take(self._stc_img, src_coord[axis], axis=axis).T
            else:
                stc_data = np.take(self._stc_img, 0, axis=axis).T * np.nan
            self._images['stc'][axis].set_data(stc_data)
            if draw:
                self._draw(axis)

    def _update_images(self, axis=None, draw=True):
        """Update images when general changes happen."""
        self._update_stc_images(axis=axis, draw=draw)
        self._update_data_plot(draw=draw)
        super()._update_images()
