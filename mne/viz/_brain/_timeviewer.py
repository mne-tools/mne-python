# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import contextlib
from functools import partial
import os
import sys
import time
import traceback
import warnings

import numpy as np
from scipy import sparse

from ._brain import _Brain
from .callback import (ShowView, IntSlider, TimeSlider, SmartSlider,
                       BumpColorbarPoints, UpdateColorbarScale)
from .mplcanvas import MplCanvas
from .view import _lh_views_dict

from ..utils import _show_help, _get_color_list
from ...externals.decorator import decorator
from ...source_space import vertex_to_mni, _read_talxfm
from ...transforms import apply_trans
from ...utils import _ReuseCycle, warn, copy_doc, _validate_type


@decorator
def safe_event(fun, *args, **kwargs):
    """Protect against PyQt5 exiting on event-handling errors."""
    try:
        return fun(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stderr)


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain, show_traces=False):
        from ..backends._pyvista import _require_minimum_version
        _require_minimum_version('0.24')

        # shared configuration
        if hasattr(brain, 'time_viewer'):
            raise RuntimeError('brain already has a TimeViewer')
        self.brain = brain
        self.orientation = list(_lh_views_dict.keys())
        self.default_smoothing_range = [0, 15]

        # detect notebook
        if brain._notebook:
            self.notebook = True
            self.configure_notebook()
            return
        else:
            self.notebook = False

        # Default configuration
        self.playback = False
        self.visibility = False
        self.refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        self.default_scaling_range = [0.2, 2.0]
        self.default_playback_speed_range = [0.01, 1]
        self.default_playback_speed_value = 0.05
        self.default_status_bar_msg = "Press ? for help"
        all_keys = ('lh', 'rh', 'vol')
        self.act_data_smooth = {key: (None, None) for key in all_keys}
        self.color_cycle = None
        self.mpl_canvas = None
        self.picked_points = {key: list() for key in all_keys}
        self.pick_table = dict()
        self._mouse_no_mvt = -1
        self.icons = dict()
        self.actions = dict()
        self.callbacks = dict()
        self.sliders = dict()
        self.keys = ('fmin', 'fmid', 'fmax')
        self.slider_length = 0.02
        self.slider_width = 0.04
        self.slider_color = (0.43137255, 0.44313725, 0.45882353)
        self.slider_tube_width = 0.04
        self.slider_tube_color = (0.69803922, 0.70196078, 0.70980392)

        # Direct access parameters:
        self.brain.time_viewer = self
        self.plotter = brain._renderer.plotter
        self.main_menu = self.plotter.main_menu
        self.window = self.plotter.app_window
        self.tool_bar = self.window.addToolBar("toolbar")
        self.status_bar = self.window.statusBar()
        self.interactor = self.plotter.interactor
        self.window.signal_close.connect(self.clean)

        # Derived parameters:
        self.playback_speed = self.default_playback_speed_value
        _validate_type(show_traces, (bool, str, 'numeric'), 'show_traces')
        self.interactor_fraction = 0.25
        if isinstance(show_traces, str):
            assert 'show_traces' == 'separate'  # should be guaranteed earlier
            self.show_traces = True
            self.separate_canvas = True
        else:
            if isinstance(show_traces, bool):
                self.show_traces = show_traces
            else:
                show_traces = float(show_traces)
                if not 0 < show_traces < 1:
                    raise ValueError(
                        'show traces, if numeric, must be between 0 and 1, '
                        f'got {show_traces}')
                self.show_traces = True
                self.interactor_fraction = show_traces
            self.separate_canvas = False
        del show_traces

        self._spheres = list()
        self.load_icons()
        self.configure_time_label()
        self.configure_sliders()
        self.configure_scalar_bar()
        self.configure_playback()
        self.configure_point_picking()
        self.configure_menu()
        self.configure_tool_bar()
        self.configure_status_bar()

        # show everything at the end
        self.toggle_interface()
        with self.ensure_minimum_sizes():
            self.brain.show()

    @contextlib.contextmanager
    def ensure_minimum_sizes(self):
        from ..backends._pyvista import _process_events
        sz = self.brain._size
        adjust_mpl = self.show_traces and not self.separate_canvas
        if not adjust_mpl:
            yield
        else:
            mpl_h = int(round((sz[1] * self.interactor_fraction) /
                              (1 - self.interactor_fraction)))
            self.mpl_canvas.canvas.setMinimumSize(sz[0], mpl_h)
            try:
                yield
            finally:
                self.splitter.setSizes([sz[1], mpl_h])
                _process_events(self.plotter)
                _process_events(self.plotter)
                self.mpl_canvas.canvas.setMinimumSize(0, 0)
            _process_events(self.plotter)
            _process_events(self.plotter)
            # sizes could change, update views
            for hemi in ('lh', 'rh'):
                for ri, ci, v in self.brain._iter_views(hemi):
                    self.brain.show_view(view=v, row=ri, col=ci)
            _process_events(self.plotter)

    def toggle_interface(self, value=None):
        if value is None:
            self.visibility = not self.visibility
        else:
            self.visibility = value

        # update tool bar icon
        if self.visibility:
            self.actions["visibility"].setIcon(self.icons["visibility_on"])
        else:
            self.actions["visibility"].setIcon(self.icons["visibility_off"])

        # manage sliders
        for slider in self.plotter.slider_widgets:
            slider_rep = slider.GetRepresentation()
            if self.visibility:
                slider_rep.VisibilityOn()
            else:
                slider_rep.VisibilityOff()

        # manage time label
        time_label = self.brain._data['time_label']
        # if we actually have time points, we will show the slider so
        # hide the time actor
        have_ts = self.brain._times is not None and len(self.brain._times) > 1
        if self.time_actor is not None:
            if self.visibility and time_label is not None and not have_ts:
                self.time_actor.SetInput(time_label(self.brain._current_time))
                self.time_actor.VisibilityOn()
            else:
                self.time_actor.VisibilityOff()

        self.plotter.update()

    def _save_movie(self, filename, **kwargs):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QCursor

        def frame_callback(frame, n_frames):
            if frame == n_frames:
                # On the ImageIO step
                self.status_msg.setText(
                    "Saving with ImageIO: %s"
                    % filename
                )
                self.status_msg.show()
                self.status_progress.hide()
                self.status_bar.layout().update()
            else:
                self.status_msg.setText(
                    "Rendering images (frame %d / %d) ..."
                    % (frame + 1, n_frames)
                )
                self.status_msg.show()
                self.status_progress.show()
                self.status_progress.setRange(0, n_frames - 1)
                self.status_progress.setValue(frame)
                self.status_progress.update()
                self.status_progress.repaint()
            self.status_msg.update()
            self.status_msg.parent().update()
            self.status_msg.repaint()

        # temporarily hide interface
        default_visibility = self.visibility
        self.toggle_interface(value=False)
        # set cursor to busy
        default_cursor = self.interactor.cursor()
        self.interactor.setCursor(QCursor(Qt.WaitCursor))

        try:
            self.brain.save_movie(
                filename=filename,
                time_dilation=(1. / self.playback_speed),
                callback=frame_callback,
                **kwargs
            )
        except (Exception, KeyboardInterrupt):
            warn('Movie saving aborted:\n' + traceback.format_exc())

        # restore visibility
        self.toggle_interface(value=default_visibility)
        # restore cursor
        self.interactor.setCursor(default_cursor)

    @copy_doc(_Brain.save_movie)
    def save_movie(self, filename=None, **kwargs):
        try:
            from pyvista.plotting.qt_plotting import FileDialog
        except ImportError:
            from pyvistaqt.plotting import FileDialog

        if filename is None:
            self.status_msg.setText("Choose movie path ...")
            self.status_msg.show()
            self.status_progress.setValue(0)

            def _clean(unused):
                del unused
                self.status_msg.hide()
                self.status_progress.hide()

            dialog = FileDialog(
                self.plotter.app_window,
                callback=partial(self._save_movie, **kwargs)
            )
            dialog.setDirectory(os.getcwd())
            dialog.finished.connect(_clean)
            return dialog
        else:
            self._save_movie(filename=filename, **kwargs)
            return

    def apply_auto_scaling(self):
        self.brain.update_auto_scaling()
        for key in ('fmin', 'fmid', 'fmax'):
            self.reps[key].SetValue(self.brain._data[key])
        self.plotter.update()

    def restore_user_scaling(self):
        self.brain.update_auto_scaling(restore=True)
        for key in ('fmin', 'fmid', 'fmax'):
            self.reps[key].SetValue(self.brain._data[key])
        self.plotter.update()

    def toggle_playback(self, value=None):
        if value is None:
            self.playback = not self.playback
        else:
            self.playback = value

        # update tool bar icon
        if self.playback:
            self.actions["play"].setIcon(self.icons["pause"])
        else:
            self.actions["play"].setIcon(self.icons["play"])

        if self.playback:
            time_data = self.brain._data['time']
            max_time = np.max(time_data)
            if self.brain._current_time == max_time:  # start over
                self.brain.set_time_point(0)  # first index
            self._last_tick = time.time()

    def reset(self):
        self.brain.reset_view()
        max_time = len(self.brain._data['time']) - 1
        if max_time > 0:
            self.callbacks["time"](
                self.brain._data["initial_time_idx"],
                update_widget=True,
            )
        self.plotter.update()

    def set_playback_speed(self, speed):
        self.playback_speed = speed

    @safe_event
    def play(self):
        if self.playback:
            try:
                self._advance()
            except Exception:
                self.toggle_playback(value=False)
                raise

    def _advance(self):
        this_time = time.time()
        delta = this_time - self._last_tick
        self._last_tick = time.time()
        time_data = self.brain._data['time']
        times = np.arange(self.brain._n_times)
        time_shift = delta * self.playback_speed
        max_time = np.max(time_data)
        time_point = min(self.brain._current_time + time_shift, max_time)
        # always use linear here -- this does not determine the data
        # interpolation mode, it just finds where we are (in time) in
        # terms of the time indices
        idx = np.interp(time_point, time_data, times)
        self.callbacks["time"](idx, update_widget=True)
        if time_point == max_time:
            self.toggle_playback(value=False)

    def set_slider_style(self):
        for slider in self.sliders.values():
            if slider is not None:
                slider_rep = slider.GetRepresentation()
                slider_rep.SetSliderLength(self.slider_length)
                slider_rep.SetSliderWidth(self.slider_width)
                slider_rep.SetTubeWidth(self.slider_tube_width)
                slider_rep.GetSliderProperty().SetColor(self.slider_color)
                slider_rep.GetTubeProperty().SetColor(self.slider_tube_color)
                slider_rep.GetLabelProperty().SetShadow(False)
                slider_rep.GetLabelProperty().SetBold(True)
                slider_rep.GetLabelProperty().SetColor(self.brain._fg_color)
                slider_rep.GetTitleProperty().ShallowCopy(
                    slider_rep.GetLabelProperty()
                )
                slider_rep.GetCapProperty().SetOpacity(0)

    def configure_notebook(self):
        from ._notebook import _NotebookInteractor
        self.brain._renderer.figure.display = _NotebookInteractor(self)

    def configure_time_label(self):
        self.time_actor = self.brain._data.get('time_actor')
        if self.time_actor is not None:
            self.time_actor.SetPosition(0.5, 0.03)
            self.time_actor.GetTextProperty().SetJustificationToCentered()
            self.time_actor.GetTextProperty().BoldOn()
            self.time_actor.VisibilityOff()

    def configure_scalar_bar(self):
        if self.brain._colorbar_added:
            scalar_bar = self.plotter.scalar_bar
            scalar_bar.SetOrientationToVertical()
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetWidth(0.05)
            scalar_bar.SetPosition(0.02, 0.2)

    def configure_sliders(self):
        # Orientation slider
        # Use 'lh' as a reference for orientation for 'both'
        if self.brain._hemi == 'both':
            hemis_ref = ['lh']
        else:
            hemis_ref = self.brain._hemis
        for hemi in hemis_ref:
            for ri, ci, view in self.brain._iter_views(hemi):
                orientation_name = f"orientation_{hemi}_{ri}_{ci}"
                self.plotter.subplot(ri, ci)
                if view == 'flat':
                    self.callbacks[orientation_name] = None
                    continue
                self.callbacks[orientation_name] = ShowView(
                    plotter=self.plotter,
                    brain=self.brain,
                    orientation=self.orientation,
                    hemi=hemi,
                    row=ri,
                    col=ci,
                )
                self.sliders[orientation_name] = \
                    self.plotter.add_text_slider_widget(
                    self.callbacks[orientation_name],
                    value=0,
                    data=self.orientation,
                    pointa=(0.82, 0.74),
                    pointb=(0.98, 0.74),
                    event_type='always'
                )
                orientation_rep = \
                    self.sliders[orientation_name].GetRepresentation()
                orientation_rep.ShowSliderLabelOff()
                self.callbacks[orientation_name].slider_rep = orientation_rep
                self.callbacks[orientation_name](view, update_widget=True)

        # Put other sliders on the bottom right view
        ri, ci = np.array(self.brain._subplot_shape) - 1
        self.plotter.subplot(ri, ci)

        # Smoothing slider
        self.callbacks["smoothing"] = IntSlider(
            plotter=self.plotter,
            callback=self.brain.set_data_smoothing,
            first_call=False,
        )
        self.sliders["smoothing"] = self.plotter.add_slider_widget(
            self.callbacks["smoothing"],
            value=self.brain._data['smoothing_steps'],
            rng=self.default_smoothing_range, title="smoothing",
            pointa=(0.82, 0.90),
            pointb=(0.98, 0.90)
        )
        self.callbacks["smoothing"].slider_rep = \
            self.sliders["smoothing"].GetRepresentation()

        # Time slider
        max_time = len(self.brain._data['time']) - 1
        # VTK on macOS bombs if we create these then hide them, so don't
        # even create them
        if max_time < 1:
            self.callbacks["time"] = None
            self.sliders["time"] = None
        else:
            self.callbacks["time"] = TimeSlider(
                plotter=self.plotter,
                brain=self.brain,
                first_call=False,
                callback=self.plot_time_line,
            )
            self.sliders["time"] = self.plotter.add_slider_widget(
                self.callbacks["time"],
                value=self.brain._data['time_idx'],
                rng=[0, max_time],
                pointa=(0.23, 0.1),
                pointb=(0.77, 0.1),
                event_type='always'
            )
            self.callbacks["time"].slider_rep = \
                self.sliders["time"].GetRepresentation()
            # configure properties of the time slider
            self.sliders["time"].GetRepresentation().SetLabelFormat(
                'idx=%0.1f')

        current_time = self.brain._current_time
        assert current_time is not None  # should never be the case, float
        time_label = self.brain._data['time_label']
        if callable(time_label):
            current_time = time_label(current_time)
        else:
            current_time = time_label
        if self.sliders["time"] is not None:
            self.sliders["time"].GetRepresentation().SetTitleText(current_time)
        if self.time_actor is not None:
            self.time_actor.SetInput(current_time)
        del current_time

        # Playback speed slider
        if self.sliders["time"] is None:
            self.callbacks["playback_speed"] = None
            self.sliders["playback_speed"] = None
        else:
            self.callbacks["playback_speed"] = SmartSlider(
                plotter=self.plotter,
                callback=self.set_playback_speed,
            )
            self.sliders["playback_speed"] = self.plotter.add_slider_widget(
                self.callbacks["playback_speed"],
                value=self.default_playback_speed_value,
                rng=self.default_playback_speed_range, title="speed",
                pointa=(0.02, 0.1),
                pointb=(0.18, 0.1),
                event_type='always'
            )
            self.callbacks["playback_speed"].slider_rep = \
                self.sliders["playback_speed"].GetRepresentation()

        # Colormap slider
        pointa = np.array((0.82, 0.26))
        pointb = np.array((0.98, 0.26))
        shift = np.array([0, 0.1])

        for idx, key in enumerate(self.keys):
            title = "clim" if not idx else ""
            rng = _get_range(self.brain)
            self.callbacks[key] = BumpColorbarPoints(
                plotter=self.plotter,
                brain=self.brain,
                name=key
            )
            self.sliders[key] = self.plotter.add_slider_widget(
                self.callbacks[key],
                value=self.brain._data[key],
                rng=rng, title=title,
                pointa=pointa + idx * shift,
                pointb=pointb + idx * shift,
                event_type="always",
            )

        # fscale
        self.callbacks["fscale"] = UpdateColorbarScale(
            plotter=self.plotter,
            brain=self.brain,
        )
        self.sliders["fscale"] = self.plotter.add_slider_widget(
            self.callbacks["fscale"],
            value=1.0,
            rng=self.default_scaling_range, title="fscale",
            pointa=(0.82, 0.10),
            pointb=(0.98, 0.10)
        )
        self.callbacks["fscale"].slider_rep = \
            self.sliders["fscale"].GetRepresentation()

        # register colorbar slider representations
        self.reps = \
            {key: self.sliders[key].GetRepresentation() for key in self.keys}
        for name in ("fmin", "fmid", "fmax", "fscale"):
            self.callbacks[name].reps = self.reps

        # set the slider style
        self.set_slider_style()

    def configure_playback(self):
        self.plotter.add_callback(self.play, self.refresh_rate_ms)

    def configure_point_picking(self):
        if not self.show_traces:
            return
        from ..backends._pyvista import _update_picking_callback
        # use a matplotlib canvas
        self.color_cycle = _ReuseCycle(_get_color_list())
        win = self.plotter.app_window
        dpi = win.windowHandle().screen().logicalDotsPerInch()
        ratio = (1 - self.interactor_fraction) / self.interactor_fraction
        w = self.interactor.geometry().width()
        h = self.interactor.geometry().height() / ratio
        # Get the fractional components for the brain and mpl
        self.mpl_canvas = MplCanvas(self, w / dpi, h / dpi, dpi)
        xlim = [np.min(self.brain._data['time']),
                np.max(self.brain._data['time'])]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.mpl_canvas.axes.set(xlim=xlim)
        if not self.separate_canvas:
            from PyQt5.QtWidgets import QSplitter
            from PyQt5.QtCore import Qt
            canvas = self.mpl_canvas.canvas
            vlayout = self.plotter.frame.layout()
            vlayout.removeWidget(self.interactor)
            self.splitter = splitter = QSplitter(
                orientation=Qt.Vertical, parent=self.plotter.frame)
            vlayout.addWidget(splitter)
            splitter.addWidget(self.interactor)
            splitter.addWidget(canvas)
        self.mpl_canvas.set_color(
            bg_color=self.brain._bg_color,
            fg_color=self.brain._fg_color,
        )
        self.mpl_canvas.show()

        # get data for each hemi
        for idx, hemi in enumerate(['vol', 'lh', 'rh']):
            hemi_data = self.brain._data.get(hemi)
            if hemi_data is not None:
                act_data = hemi_data['array']
                if act_data.ndim == 3:
                    act_data = np.linalg.norm(act_data, axis=1)
                smooth_mat = hemi_data.get('smooth_mat')
                vertices = hemi_data['vertices']
                if hemi == 'vol':
                    assert smooth_mat is None
                    smooth_mat = sparse.csr_matrix(
                        (np.ones(len(vertices)),
                         (vertices, np.arange(len(vertices)))))
                self.act_data_smooth[hemi] = (act_data, smooth_mat)

        # plot the GFP
        y = np.concatenate(list(v[0] for v in self.act_data_smooth.values()
                                if v[0] is not None))
        y = np.linalg.norm(y, axis=0) / np.sqrt(len(y))
        self.mpl_canvas.axes.plot(
            self.brain._data['time'], y,
            lw=3, label='GFP', zorder=3, color=self.brain._fg_color,
            alpha=0.5, ls=':')

        # now plot the time line
        self.plot_time_line()

        # then the picked points
        for idx, hemi in enumerate(['lh', 'rh', 'vol']):
            act_data = self.act_data_smooth.get(hemi, [None])[0]
            if act_data is None:
                continue
            hemi_data = self.brain._data[hemi]
            vertices = hemi_data['vertices']

            # simulate a picked renderer
            if self.brain._hemi in ('both', 'rh') or hemi == 'vol':
                idx = 0
            self.picked_renderer = self.plotter.renderers[idx]

            # initialize the default point
            if self.brain._data['initial_time'] is not None:
                # pick at that time
                use_data = act_data[
                    :, [np.round(self.brain._data['time_idx']).astype(int)]]
            else:
                use_data = act_data
            ind = np.unravel_index(np.argmax(np.abs(use_data), axis=None),
                                   use_data.shape)
            if hemi == 'vol':
                mesh = hemi_data['grid']
            else:
                mesh = hemi_data['mesh']
            vertex_id = vertices[ind[0]]
            self.add_point(hemi, mesh, vertex_id)

        _update_picking_callback(
            self.plotter,
            self.on_mouse_move,
            self.on_button_press,
            self.on_button_release,
            self.on_pick
        )

    def load_icons(self):
        from PyQt5.QtGui import QIcon
        from ..backends._pyvista import _init_resources
        _init_resources()
        self.icons["help"] = QIcon(":/help.svg")
        self.icons["play"] = QIcon(":/play.svg")
        self.icons["pause"] = QIcon(":/pause.svg")
        self.icons["reset"] = QIcon(":/reset.svg")
        self.icons["scale"] = QIcon(":/scale.svg")
        self.icons["clear"] = QIcon(":/clear.svg")
        self.icons["movie"] = QIcon(":/movie.svg")
        self.icons["restore"] = QIcon(":/restore.svg")
        self.icons["screenshot"] = QIcon(":/screenshot.svg")
        self.icons["visibility_on"] = QIcon(":/visibility_on.svg")
        self.icons["visibility_off"] = QIcon(":/visibility_off.svg")

    def configure_tool_bar(self):
        self.actions["screenshot"] = self.tool_bar.addAction(
            self.icons["screenshot"],
            "Take a screenshot",
            self.plotter._qt_screenshot
        )
        self.actions["movie"] = self.tool_bar.addAction(
            self.icons["movie"],
            "Save movie...",
            self.save_movie
        )
        self.actions["visibility"] = self.tool_bar.addAction(
            self.icons["visibility_on"],
            "Toggle Visibility",
            self.toggle_interface
        )
        self.actions["play"] = self.tool_bar.addAction(
            self.icons["play"],
            "Play/Pause",
            self.toggle_playback
        )
        self.actions["reset"] = self.tool_bar.addAction(
            self.icons["reset"],
            "Reset",
            self.reset
        )
        self.actions["scale"] = self.tool_bar.addAction(
            self.icons["scale"],
            "Auto-Scale",
            self.apply_auto_scaling
        )
        self.actions["restore"] = self.tool_bar.addAction(
            self.icons["restore"],
            "Restore scaling",
            self.restore_user_scaling
        )
        self.actions["clear"] = self.tool_bar.addAction(
            self.icons["clear"],
            "Clear traces",
            self.clear_points
        )
        self.actions["help"] = self.tool_bar.addAction(
            self.icons["help"],
            "Help",
            self.help
        )

        self.actions["movie"].setShortcut("ctrl+shift+s")
        self.actions["visibility"].setShortcut("i")
        self.actions["play"].setShortcut(" ")
        self.actions["scale"].setShortcut("s")
        self.actions["restore"].setShortcut("r")
        self.actions["clear"].setShortcut("c")
        self.actions["help"].setShortcut("?")

    def configure_menu(self):
        # remove default picking menu
        to_remove = list()
        for action in self.main_menu.actions():
            if action.text() == "Tools":
                to_remove.append(action)
        for action in to_remove:
            self.main_menu.removeAction(action)

        # add help menu
        menu = self.main_menu.addMenu('Help')
        menu.addAction('Show MNE key bindings\t?', self.help)

    def configure_status_bar(self):
        from PyQt5.QtWidgets import QLabel, QProgressBar
        self.status_msg = QLabel(self.default_status_bar_msg)
        self.status_progress = QProgressBar()
        self.status_bar.layout().addWidget(self.status_msg, 1)
        self.status_bar.layout().addWidget(self.status_progress, 0)
        self.status_progress.hide()

    def on_mouse_move(self, vtk_picker, event):
        if self._mouse_no_mvt:
            self._mouse_no_mvt -= 1

    def on_button_press(self, vtk_picker, event):
        self._mouse_no_mvt = 2

    def on_button_release(self, vtk_picker, event):
        if self._mouse_no_mvt > 0:
            x, y = vtk_picker.GetEventPosition()
            # programmatically detect the picked renderer
            self.picked_renderer = self.plotter.iren.FindPokedRenderer(x, y)
            # trigger the pick
            self.plotter.picker.Pick(x, y, 0, self.picked_renderer)
        self._mouse_no_mvt = 0

    def on_pick(self, vtk_picker, event):
        # vtk_picker is a vtkCellPicker
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()

        if mesh is None or cell_id == -1 or not self._mouse_no_mvt:
            return  # don't pick

        # 1) Check to see if there are any spheres along the ray
        if len(self._spheres):
            collection = vtk_picker.GetProp3Ds()
            found_sphere = None
            for ii in range(collection.GetNumberOfItems()):
                actor = collection.GetItemAsObject(ii)
                for sphere in self._spheres:
                    if any(a is actor for a in sphere._actors):
                        found_sphere = sphere
                        break
                if found_sphere is not None:
                    break
            if found_sphere is not None:
                assert found_sphere._is_point
                mesh = found_sphere

        # 2) Remove sphere if it's what we have
        if hasattr(mesh, "_is_point"):
            self.remove_point(mesh)
            return

        # 3) Otherwise, pick the objects in the scene
        try:
            hemi = mesh._hemi
        except AttributeError:  # volume
            hemi = 'vol'
        else:
            assert hemi in ('lh', 'rh')
        if self.act_data_smooth[hemi][0] is None:  # no data to add for hemi
            return
        pos = np.array(vtk_picker.GetPickPosition())
        if hemi == 'vol':
            # VTK will give us the point closest to the viewer in the vol.
            # We want to pick the point with the maximum value along the
            # camera-to-click array, which fortunately we can get "just"
            # by inspecting the points that are sufficiently close to the
            # ray.
            grid = mesh = self.brain._data[hemi]['grid']
            vertices = self.brain._data[hemi]['vertices']
            coords = self.brain._data[hemi]['grid_coords'][vertices]
            scalars = grid.cell_arrays['values'][vertices]
            spacing = np.array(grid.GetSpacing())
            max_dist = np.linalg.norm(spacing) / 2.
            origin = vtk_picker.GetRenderer().GetActiveCamera().GetPosition()
            ori = pos - origin
            ori /= np.linalg.norm(ori)
            # the magic formula: distance from a ray to a given point
            dists = np.linalg.norm(np.cross(ori, coords - pos), axis=1)
            assert dists.shape == (len(coords),)
            mask = dists <= max_dist
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return  # weird point on edge of volume?
            # useful for debugging the ray by mapping it into the volume:
            # dists = dists - dists.min()
            # dists = (1. - dists / dists.max()) * self.brain._cmap_range[1]
            # grid.cell_arrays['values'][vertices] = dists * mask
            idx = idx[np.argmax(np.abs(scalars[idx]))]
            vertex_id = vertices[idx]
            # Naive way: convert pos directly to idx; i.e., apply mri_src_t
            # shape = self.brain._data[hemi]['grid_shape']
            # taking into account the cell vs point difference (spacing/2)
            # shift = np.array(grid.GetOrigin()) + spacing / 2.
            # ijk = np.round((pos - shift) / spacing).astype(int)
            # vertex_id = np.ravel_multi_index(ijk, shape, order='F')
        else:
            vtk_cell = mesh.GetCell(cell_id)
            cell = [vtk_cell.GetPointId(point_id) for point_id
                    in range(vtk_cell.GetNumberOfPoints())]
            vertices = mesh.points[cell]
            idx = np.argmin(abs(vertices - pos), axis=0)
            vertex_id = cell[idx[0]]

        if vertex_id not in self.picked_points[hemi]:
            self.add_point(hemi, mesh, vertex_id)

    def add_point(self, hemi, mesh, vertex_id):
        # skip if the wrong hemi is selected
        if self.act_data_smooth[hemi][0] is None:
            return
        from ..backends._pyvista import _sphere
        color = next(self.color_cycle)
        line = self.plot_time_course(hemi, vertex_id, color)
        if hemi == 'vol':
            ijk = np.unravel_index(
                vertex_id, np.array(mesh.GetDimensions()) - 1, order='F')
            # should just be GetCentroid(center), but apparently it's VTK9+:
            # center = np.empty(3)
            # voxel.GetCentroid(center)
            voxel = mesh.GetCell(*ijk)
            pts = voxel.GetPoints()
            n_pts = pts.GetNumberOfPoints()
            center = np.empty((n_pts, 3))
            for ii in range(pts.GetNumberOfPoints()):
                pts.GetPoint(ii, center[ii])
            center = np.mean(center, axis=0)
        else:
            center = mesh.GetPoints().GetPoint(vertex_id)
        del mesh

        # from the picked renderer to the subplot coords
        rindex = self.plotter.renderers.index(self.picked_renderer)
        row, col = self.plotter.index_to_loc(rindex)

        actors = list()
        spheres = list()
        for ri, ci, _ in self.brain._iter_views(hemi):
            self.plotter.subplot(ri, ci)
            # Using _sphere() instead of renderer.sphere() for 2 reasons:
            # 1) renderer.sphere() fails on Windows in a scenario where a lot
            #    of picking requests are done in a short span of time (could be
            #    mitigated with synchronization/delay?)
            # 2) the glyph filter is used in renderer.sphere() but only one
            #    sphere is required in this function.
            actor, sphere = _sphere(
                plotter=self.plotter,
                center=np.array(center),
                color=color,
                radius=4.0,
            )
            actors.append(actor)
            spheres.append(sphere)

        # add metadata for picking
        for sphere in spheres:
            sphere._is_point = True
            sphere._hemi = hemi
            sphere._line = line
            sphere._actors = actors
            sphere._color = color
            sphere._vertex_id = vertex_id

        self.picked_points[hemi].append(vertex_id)
        self._spheres.extend(spheres)
        self.pick_table[vertex_id] = spheres

    def remove_point(self, mesh):
        vertex_id = mesh._vertex_id
        if vertex_id not in self.pick_table:
            return

        hemi = mesh._hemi
        color = mesh._color
        spheres = self.pick_table[vertex_id]
        spheres[0]._line.remove()
        self.mpl_canvas.update_plot()
        self.picked_points[hemi].remove(vertex_id)

        with warnings.catch_warnings(record=True):
            # We intentionally ignore these in case we have traversed the
            # entire color cycle
            warnings.simplefilter('ignore')
            self.color_cycle.restore(color)
        for sphere in spheres:
            # remove all actors
            self.plotter.remove_actor(sphere._actors)
            sphere._actors = None
            self._spheres.pop(self._spheres.index(sphere))
        self.pick_table.pop(vertex_id)

    def clear_points(self):
        for sphere in list(self._spheres):  # will remove itself, so copy
            self.remove_point(sphere)
        assert sum(len(v) for v in self.picked_points.values()) == 0
        assert len(self.pick_table) == 0
        assert len(self._spheres) == 0

    def plot_time_course(self, hemi, vertex_id, color):
        if self.mpl_canvas is None:
            return
        time = self.brain._data['time'].copy()  # avoid circular ref
        if hemi == 'vol':
            hemi_str = 'V'
            xfm = _read_talxfm(
                self.brain._subject_id, self.brain._subjects_dir)
            if self.brain._units == 'm':
                xfm['trans'][:3, 3] /= 1000.
            ijk = np.unravel_index(
                vertex_id, self.brain._data[hemi]['grid_shape'], order='F')
            src_mri_t = self.brain._data[hemi]['grid_src_mri_t']
            mni = apply_trans(np.dot(xfm['trans'], src_mri_t), ijk)
        else:
            hemi_str = 'L' if hemi == 'lh' else 'R'
            mni = vertex_to_mni(
                vertices=vertex_id,
                hemis=0 if hemi == 'lh' else 1,
                subject=self.brain._subject_id,
                subjects_dir=self.brain._subjects_dir
            )
        label = "{}:{} MNI: {}".format(
            hemi_str, str(vertex_id).ljust(6),
            ', '.join('%5.1f' % m for m in mni))
        act_data, smooth = self.act_data_smooth[hemi]
        if smooth is not None:
            act_data = smooth[vertex_id].dot(act_data)[0]
        else:
            act_data = act_data[vertex_id].copy()
        line = self.mpl_canvas.plot(
            time,
            act_data,
            label=label,
            lw=1.,
            color=color,
            zorder=4,
        )
        return line

    def plot_time_line(self):
        if self.mpl_canvas is None:
            return
        if isinstance(self.show_traces, bool) and self.show_traces:
            # add time information
            current_time = self.brain._current_time
            if not hasattr(self, "time_line"):
                self.time_line = self.mpl_canvas.plot_time_line(
                    x=current_time,
                    label='time',
                    color=self.brain._fg_color,
                    lw=1,
                )
            self.time_line.set_xdata(current_time)
            self.mpl_canvas.update_plot()

    def help(self):
        pairs = [
            ('?', 'Display help window'),
            ('i', 'Toggle interface'),
            ('s', 'Apply auto-scaling'),
            ('r', 'Restore original clim'),
            ('c', 'Clear all traces'),
            ('Space', 'Start/Pause playback'),
        ]
        text1, text2 = zip(*pairs)
        text1 = '\n'.join(text1)
        text2 = '\n'.join(text2)
        _show_help(
            col1=text1,
            col2=text2,
            width=5,
            height=2,
        )

    def clear_callbacks(self):
        for callback in self.callbacks.values():
            if callback is not None:
                if hasattr(callback, "plotter"):
                    callback.plotter = None
                if hasattr(callback, "brain"):
                    callback.brain = None
                if hasattr(callback, "slider_rep"):
                    callback.slider_rep = None
        self.callbacks.clear()

    @safe_event
    def clean(self):
        # resolve the reference cycle
        self.clear_points()
        self.clear_callbacks()
        self.actions.clear()
        self.sliders.clear()
        self.reps = None
        self.brain.time_viewer = None
        self.brain = None
        self.plotter = None
        self.main_menu = None
        self.window = None
        self.tool_bar = None
        self.status_bar = None
        self.interactor = None
        if self.mpl_canvas is not None:
            self.mpl_canvas.clear()
            self.mpl_canvas = None
        self.time_actor = None
        self.picked_renderer = None
        for key in list(self.act_data_smooth.keys()):
            self.act_data_smooth[key] = None


def _get_range(brain):
    val = np.abs(np.concatenate(list(brain._current_act_data.values())))
    return [np.min(val), np.max(val)]


def _normalize(point, shape):
    return (point[0] / shape[1], point[1] / shape[0])
