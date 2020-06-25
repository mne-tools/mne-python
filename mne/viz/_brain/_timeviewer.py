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

from . import _Brain
from ..utils import _check_option, _show_help, _get_color_list, tight_layout
from ...externals.decorator import decorator
from ...source_space import vertex_to_mni
from ...utils import _ReuseCycle, warn, copy_doc


@decorator
def run_once(fun, *args, **kwargs):
    """Run the function only once."""
    if not hasattr(fun, "_has_run"):
        fun._has_run = True
        return fun(*args, **kwargs)


@decorator
def safe_event(fun, *args, **kwargs):
    """Protect against PyQt5 exiting on event-handling errors."""
    try:
        return fun(*args, **kwargs)
    except Exception:
        traceback.print_exc(file=sys.stderr)


class MplCanvas(object):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, time_viewer, width, height, dpi):
        from PyQt5 import QtWidgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        if time_viewer.separate_canvas:
            parent = None
        else:
            parent = time_viewer.window
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.set(xlabel='Time (sec)', ylabel='Activation (AU)')
        self.canvas.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self.canvas,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self.canvas)
        # XXX eventually this should be called in the window resize callback
        tight_layout(fig=self.axes.figure)
        self.time_viewer = time_viewer
        for event in ('button_press', 'motion_notify'):
            self.canvas.mpl_connect(
                event + '_event', getattr(self, 'on_' + event))

    def plot(self, x, y, label, **kwargs):
        """Plot a curve."""
        line, = self.axes.plot(
            x, y, label=label, **kwargs)
        self.update_plot()
        return line

    def plot_time_line(self, x, label, **kwargs):
        """Plot the vertical line."""
        line = self.axes.axvline(x, label=label, **kwargs)
        self.update_plot()
        return line

    def update_plot(self):
        """Update the plot."""
        self.axes.legend(prop={'family': 'monospace', 'size': 'small'},
                         framealpha=0.5, handlelength=1.)
        self.canvas.draw()

    def set_color(self, bg_color, fg_color):
        """Set the widget colors."""
        self.axes.set_facecolor(bg_color)
        self.axes.xaxis.label.set_color(fg_color)
        self.axes.yaxis.label.set_color(fg_color)
        self.axes.spines['top'].set_color(fg_color)
        self.axes.spines['bottom'].set_color(fg_color)
        self.axes.spines['left'].set_color(fg_color)
        self.axes.spines['right'].set_color(fg_color)
        self.axes.tick_params(axis='x', colors=fg_color)
        self.axes.tick_params(axis='y', colors=fg_color)
        self.fig.patch.set_facecolor(bg_color)

    def show(self):
        """Show the canvas."""
        self.canvas.show()

    def close(self):
        """Close the canvas."""
        self.canvas.close()

    def on_button_press(self, event):
        """Handle button presses."""
        # left click (and maybe drag) in progress in axes
        if (event.inaxes != self.axes or
                event.button != 1):
            return
        self.time_viewer.time_call(
            event.xdata, update_widget=True, time_as_index=False)

    on_motion_notify = on_button_press  # for now they can be the same


class IntSlider(object):
    """Class to set a integer slider."""

    def __init__(self, plotter=None, callback=None, first_call=True):
        self.plotter = plotter
        self.callback = callback
        self.slider_rep = None
        self.first_call = first_call
        self._first_time = True

    def __call__(self, value):
        """Round the label of the slider."""
        idx = int(round(value))
        if self.slider_rep is not None:
            self.slider_rep.SetValue(idx)
            self.plotter.update()
        if not self._first_time or all([self._first_time, self.first_call]):
            self.callback(idx)
        if self._first_time:
            self._first_time = False


class TimeSlider(object):
    """Class to update the time slider."""

    def __init__(self, plotter=None, brain=None, callback=None,
                 first_call=True):
        self.plotter = plotter
        self.brain = brain
        self.callback = callback
        self.slider_rep = None
        self.first_call = first_call
        self._first_time = True
        self.time_label = None
        if self.brain is not None and callable(self.brain._data['time_label']):
            self.time_label = self.brain._data['time_label']

    def __call__(self, value, update_widget=False, time_as_index=True):
        """Update the time slider."""
        value = float(value)
        if not time_as_index:
            value = self.brain._to_time_index(value)
        if not self._first_time or all([self._first_time, self.first_call]):
            self.brain.set_time_point(value)
        if self.callback is not None:
            self.callback()
        current_time = self.brain._current_time
        if self.slider_rep is not None:
            if self.time_label is not None:
                current_time = self.time_label(current_time)
                self.slider_rep.SetTitleText(current_time)
            if update_widget:
                self.slider_rep.SetValue(value)
                self.plotter.update()
        if self._first_time:
            self._first_time = False


class UpdateColorbarScale(object):
    """Class to update the values of the colorbar sliders."""

    def __init__(self, plotter=None, brain=None):
        self.plotter = plotter
        self.brain = brain
        self.keys = ('fmin', 'fmid', 'fmax')
        self.reps = {key: None for key in self.keys}
        self.fscale_slider_rep = None

    def __call__(self, value):
        """Update the colorbar sliders."""
        self.brain.update_fscale(value)
        for key in self.keys:
            if self.reps[key] is not None:
                self.reps[key].SetValue(self.brain._data[key])
        if self.fscale_slider_rep is not None:
            self.fscale_slider_rep.SetValue(1.0)
        self.plotter.update()


class BumpColorbarPoints(object):
    """Class that ensure constraints over the colorbar points."""

    def __init__(self, plotter=None, brain=None, name=None):
        self.plotter = plotter
        self.brain = brain
        self.name = name
        self.callback = {
            "fmin": brain.update_fmin,
            "fmid": brain.update_fmid,
            "fmax": brain.update_fmax
        }
        self.keys = ('fmin', 'fmid', 'fmax')
        self.reps = {key: None for key in self.keys}
        self.last_update = time.time()

    def __call__(self, value):
        """Update the colorbar sliders."""
        vals = {key: self.brain._data[key] for key in self.keys}
        if self.name == "fmin" and self.reps["fmin"] is not None:
            if vals['fmax'] < value:
                self.brain.update_fmax(value)
                self.reps['fmax'].SetValue(value)
            if vals['fmid'] < value:
                self.brain.update_fmid(value)
                self.reps['fmid'].SetValue(value)
            self.reps['fmin'].SetValue(value)
        elif self.name == "fmid" and self.reps['fmid'] is not None:
            if vals['fmin'] > value:
                self.brain.update_fmin(value)
                self.reps['fmin'].SetValue(value)
            if vals['fmax'] < value:
                self.brain.update_fmax(value)
                self.reps['fmax'].SetValue(value)
            self.reps['fmid'].SetValue(value)
        elif self.name == "fmax" and self.reps['fmax'] is not None:
            if vals['fmin'] > value:
                self.brain.update_fmin(value)
                self.reps['fmin'].SetValue(value)
            if vals['fmid'] > value:
                self.brain.update_fmid(value)
                self.reps['fmid'].SetValue(value)
            self.reps['fmax'].SetValue(value)
        if time.time() > self.last_update + 1. / 60.:
            self.callback[self.name](value)
            self.last_update = time.time()
        self.plotter.update()


class ShowView(object):
    """Class that selects the correct view."""

    def __init__(self, plotter=None, brain=None, orientation=None,
                 row=None, col=None, hemi=None):
        self.plotter = plotter
        self.brain = brain
        self.orientation = orientation
        self.short_orientation = [s[:3] for s in orientation]
        self.row = row
        self.col = col
        self.hemi = hemi
        self.slider_rep = None

    def __call__(self, value, update_widget=False):
        """Update the view."""
        self.brain.show_view(value, row=self.row, col=self.col,
                             hemi=self.hemi)
        if update_widget:
            if len(value) > 3:
                idx = self.orientation.index(value)
            else:
                idx = self.short_orientation.index(value)
            if self.slider_rep is not None:
                self.slider_rep.SetValue(idx)
                self.slider_rep.SetTitleText(self.orientation[idx])
                self.plotter.update()


class SmartSlider(object):
    """Class to manage smart slider.

    It stores it's own slider representation for efficiency
    and uses it when necessary.
    """

    def __init__(self, plotter=None, callback=None):
        self.plotter = plotter
        self.callback = callback
        self.slider_rep = None

    def __call__(self, value, update_widget=False):
        """Update the value."""
        self.callback(value)
        if update_widget:
            if self.slider_rep is not None:
                self.slider_rep.SetValue(value)
                self.plotter.update()


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain, show_traces=False):
        from ..backends._pyvista import _require_minimum_version
        _require_minimum_version('0.24')

        # shared configuration
        self.brain = brain
        self.orientation = [
            'lateral',
            'medial',
            'rostral',
            'caudal',
            'dorsal',
            'ventral',
            'frontal',
            'parietal'
        ]
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
        self.act_data = {'lh': None, 'rh': None}
        self.color_cycle = None
        self.picked_points = {'lh': list(), 'rh': list()}
        self._mouse_no_mvt = -1
        self.icons = dict()
        self.actions = dict()
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
        _check_option('show_traces', type(show_traces), [bool, str])
        if isinstance(show_traces, str) and show_traces == "separate":
            self.show_traces = True
            self.separate_canvas = True
        else:
            self.show_traces = show_traces
            self.separate_canvas = False

        self.load_icons()
        self.interactor_stretch = 3
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
        sz = self.brain._size
        adjust_mpl = self.show_traces and not self.separate_canvas
        if not adjust_mpl:
            yield
        else:
            self.mpl_canvas.canvas.setMinimumSize(
                sz[0], int(round(sz[1] / self.interactor_stretch)))
            try:
                yield
            finally:
                self.mpl_canvas.canvas.setMinimumSize(0, 0)

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
        from pyvista.plotting.qt_plotting import FileDialog

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
        self.time_call(idx, update_widget=True)
        if time_point == max_time:
            self.toggle_playback(value=False)

    def set_slider_style(self, slider, show_label=True, show_cap=False):
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
            if not show_cap:
                slider_rep.GetCapProperty().SetOpacity(0)
            if not show_label:
                slider_rep.ShowSliderLabelOff()

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
        rng = _get_range(self.brain)
        # Orientation slider
        # default: put orientation slider on the first view
        if self.brain._hemi in ('split', 'both'):
            self.plotter.subplot(0, 0)

        # Use 'lh' as a reference for orientation for 'both'
        if self.brain._hemi == 'both':
            hemis_ref = ['lh']
        else:
            hemis_ref = self.brain._hemis
        for hemi in hemis_ref:
            if self.brain._hemi == 'split':
                ci = 0 if hemi == 'lh' else 1
            else:
                ci = 0
            for ri, view in enumerate(self.brain._views):
                self.plotter.subplot(ri, ci)
                self.orientation_call = ShowView(
                    plotter=self.plotter,
                    brain=self.brain,
                    orientation=self.orientation,
                    hemi=hemi,
                    row=ri,
                    col=ci,
                )
                orientation_slider = self.plotter.add_text_slider_widget(
                    self.orientation_call,
                    value=0,
                    data=self.orientation,
                    pointa=(0.82, 0.74),
                    pointb=(0.98, 0.74),
                    event_type='always'
                )
                self.orientation_call.slider_rep = \
                    orientation_slider.GetRepresentation()
                self.set_slider_style(orientation_slider, show_label=False)
                self.orientation_call(view, update_widget=True)

        # necessary because show_view modified subplot
        if self.brain._hemi in ('split', 'both'):
            self.plotter.subplot(0, 0)

        # Smoothing slider
        self.smoothing_call = IntSlider(
            plotter=self.plotter,
            callback=self.brain.set_data_smoothing,
            first_call=False,
        )
        smoothing_slider = self.plotter.add_slider_widget(
            self.smoothing_call,
            value=self.brain._data['smoothing_steps'],
            rng=self.default_smoothing_range, title="smoothing",
            pointa=(0.82, 0.90),
            pointb=(0.98, 0.90)
        )
        self.smoothing_call.slider_rep = smoothing_slider.GetRepresentation()

        # Time slider
        max_time = len(self.brain._data['time']) - 1
        # VTK on macOS bombs if we create these then hide them, so don't
        # even create them
        if max_time < 1:
            self.time_call = None
            time_slider = None
        else:
            self.time_call = TimeSlider(
                plotter=self.plotter,
                brain=self.brain,
                first_call=False,
                callback=self.plot_time_line,
            )
            time_slider = self.plotter.add_slider_widget(
                self.time_call,
                value=self.brain._data['time_idx'],
                rng=[0, max_time],
                pointa=(0.23, 0.1),
                pointb=(0.77, 0.1),
                event_type='always'
            )
            self.time_call.slider_rep = time_slider.GetRepresentation()
            # configure properties of the time slider
            time_slider.GetRepresentation().SetLabelFormat('idx=%0.1f')

        current_time = self.brain._current_time
        assert current_time is not None  # should never be the case, float
        time_label = self.brain._data['time_label']
        if callable(time_label):
            current_time = time_label(current_time)
        else:
            current_time = time_label
        if time_slider is not None:
            time_slider.GetRepresentation().SetTitleText(current_time)
        if self.time_actor is not None:
            self.time_actor.SetInput(current_time)
        del current_time

        # Playback speed slider
        if time_slider is None:
            self.playback_speed_call = None
            playback_speed_slider = None
        else:
            self.playback_speed_call = SmartSlider(
                plotter=self.plotter,
                callback=self.set_playback_speed,
            )
            playback_speed_slider = self.plotter.add_slider_widget(
                self.playback_speed_call,
                value=self.default_playback_speed_value,
                rng=self.default_playback_speed_range, title="speed",
                pointa=(0.02, 0.1),
                pointb=(0.18, 0.1),
                event_type='always'
            )
            self.playback_speed_call.slider_rep = \
                playback_speed_slider.GetRepresentation()

        # Colormap slider
        pointa = np.array((0.82, 0.26))
        pointb = np.array((0.98, 0.26))
        shift = np.array([0, 0.1])
        # fmin
        self.fmin_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=self.brain,
            name="fmin"
        )
        fmin_slider = self.plotter.add_slider_widget(
            self.fmin_call,
            value=self.brain._data["fmin"],
            rng=rng, title="clim",
            pointa=pointa,
            pointb=pointb,
            event_type="always",
        )
        # fmid
        self.fmid_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=self.brain,
            name="fmid",
        )
        fmid_slider = self.plotter.add_slider_widget(
            self.fmid_call,
            value=self.brain._data["fmid"],
            rng=rng, title="",
            pointa=pointa + shift,
            pointb=pointb + shift,
            event_type="always",
        )
        # fmax
        self.fmax_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=self.brain,
            name="fmax",
        )
        fmax_slider = self.plotter.add_slider_widget(
            self.fmax_call,
            value=self.brain._data["fmax"],
            rng=rng, title="",
            pointa=pointa + 2 * shift,
            pointb=pointb + 2 * shift,
            event_type="always",
        )
        # fscale
        self.fscale_call = UpdateColorbarScale(
            plotter=self.plotter,
            brain=self.brain,
        )
        fscale_slider = self.plotter.add_slider_widget(
            self.fscale_call,
            value=1.0,
            rng=self.default_scaling_range, title="fscale",
            pointa=(0.82, 0.10),
            pointb=(0.98, 0.10)
        )
        self.fscale_call.fscale_slider_rep = fscale_slider.GetRepresentation()

        # register colorbar slider representations
        self.reps = {
            "fmin": fmin_slider.GetRepresentation(),
            "fmid": fmid_slider.GetRepresentation(),
            "fmax": fmax_slider.GetRepresentation(),
        }
        self.fmin_call.reps = self.reps
        self.fmid_call.reps = self.reps
        self.fmax_call.reps = self.reps
        self.fscale_call.reps = self.reps

        # set the slider style
        self.set_slider_style(smoothing_slider)
        self.set_slider_style(fmin_slider)
        self.set_slider_style(fmid_slider)
        self.set_slider_style(fmax_slider)
        self.set_slider_style(fscale_slider)
        if time_slider is not None:
            self.set_slider_style(playback_speed_slider)
            self.set_slider_style(time_slider)

        # store sliders for linking
        self._time_slider = time_slider
        self._playback_speed_slider = playback_speed_slider

    def configure_playback(self):
        self.plotter.add_callback(self.play, self.refresh_rate_ms)

    def configure_point_picking(self):
        from ..backends._pyvista import _update_picking_callback
        if self.show_traces:
            # use a matplotlib canvas
            self.color_cycle = _ReuseCycle(_get_color_list())
            win = self.plotter.app_window
            dpi = win.windowHandle().screen().logicalDotsPerInch()
            w, h = win.geometry().width() / dpi, win.geometry().height() / dpi
            h /= 3  # one third of the window
            self.mpl_canvas = MplCanvas(self, w, h, dpi)
            xlim = [np.min(self.brain._data['time']),
                    np.max(self.brain._data['time'])]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.mpl_canvas.axes.set(xlim=xlim)
            vlayout = self.plotter.frame.layout()
            if not self.separate_canvas:
                vlayout.addWidget(self.mpl_canvas.canvas)
                vlayout.setStretch(0, self.interactor_stretch)
                vlayout.setStretch(1, 1)
            self.mpl_canvas.set_color(
                bg_color=self.brain._bg_color,
                fg_color=self.brain._fg_color,
            )
            self.mpl_canvas.show()

            # get brain data
            for idx, hemi in enumerate(['lh', 'rh']):
                hemi_data = self.brain._data.get(hemi)
                if hemi_data is not None:
                    act_data = hemi_data['array']
                    if act_data.ndim == 3:
                        act_data = np.linalg.norm(act_data, axis=1)
                    smooth_mat = hemi_data['smooth_mat']
                    if smooth_mat is not None:
                        act_data = smooth_mat.dot(act_data)
                    self.act_data[hemi] = act_data

                    # simulate a picked renderer
                    if self.brain._hemi == 'split':
                        self.picked_renderer = self.plotter.renderers[idx]
                    else:
                        self.picked_renderer = self.plotter.renderers[0]

                    # initialize the default point
                    color = next(self.color_cycle)
                    ind = np.unravel_index(
                        np.argmax(self.act_data[hemi], axis=None),
                        self.act_data[hemi].shape
                    )
                    vertex_id = ind[0]
                    mesh = hemi_data['mesh'][-1]
                    line = self.plot_time_course(hemi, vertex_id, color)
                    self.add_point(hemi, mesh, vertex_id, line, color)

            self.plot_time_line()

            _update_picking_callback(
                self.plotter,
                self.on_mouse_move,
                self.on_button_press,
                self.on_button_release,
                self.on_pick
            )

    def load_icons(self):
        from PyQt5.QtGui import QIcon
        _init_resources()
        self.icons["help"] = QIcon(":/help.svg")
        self.icons["play"] = QIcon(":/play.svg")
        self.icons["pause"] = QIcon(":/pause.svg")
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
        cell_id = vtk_picker.GetCellId()
        mesh = vtk_picker.GetDataSet()

        if mesh is None or cell_id == -1:
            return

        if hasattr(mesh, "_is_point"):
            self.remove_point(mesh)
        elif self._mouse_no_mvt:
            hemi = mesh._hemi
            pos = vtk_picker.GetPickPosition()
            vtk_cell = mesh.GetCell(cell_id)
            cell = [vtk_cell.GetPointId(point_id) for point_id
                    in range(vtk_cell.GetNumberOfPoints())]
            vertices = mesh.points[cell]
            idx = np.argmin(abs(vertices - pos), axis=0)
            vertex_id = cell[idx[0]]

            if vertex_id not in self.picked_points[hemi]:
                color = next(self.color_cycle)

                # update associated time course
                line = self.plot_time_course(hemi, vertex_id, color)

                # add glyph at picked point
                self.add_point(hemi, mesh, vertex_id, line, color)

    def add_point(self, hemi, mesh, vertex_id, line, color):
        from ..backends._pyvista import _sphere
        center = mesh.GetPoints().GetPoint(vertex_id)

        # from the picked renderer to the subplot coords
        rindex = self.plotter.renderers.index(self.picked_renderer)
        row, col = self.plotter.index_to_loc(rindex)

        actors = list()
        spheres = list()
        for ri, view in enumerate(self.brain._views):
            self.plotter.subplot(ri, col)
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

        # this is used for testing only
        if hasattr(self, "_spheres"):
            self._spheres += spheres
        else:
            self._spheres = spheres

    def remove_point(self, mesh):
        mesh._line.remove()
        self.mpl_canvas.update_plot()
        self.picked_points[mesh._hemi].remove(mesh._vertex_id)
        with warnings.catch_warnings(record=True):
            # We intentionally ignore these in case we have traversed the
            # entire color cycle
            warnings.simplefilter('ignore')
            self.color_cycle.restore(mesh._color)
        self.plotter.remove_actor(mesh._actors)
        mesh._actors = None

    def clear_points(self):
        if hasattr(self, "_spheres"):
            for sphere in self._spheres:
                vertex_id = sphere._vertex_id
                hemi = sphere._hemi
                if vertex_id in self.picked_points[hemi]:
                    self.remove_point(sphere)
            self._spheres.clear()

    def plot_time_course(self, hemi, vertex_id, color):
        if not hasattr(self, "mpl_canvas"):
            return
        time = self.brain._data['time']
        hemi_str = 'L' if hemi == 'lh' else 'R'
        hemi_int = 0 if hemi == 'lh' else 1
        mni = vertex_to_mni(
            vertices=vertex_id,
            hemis=hemi_int,
            subject=self.brain._subject_id,
            subjects_dir=self.brain._subjects_dir
        )
        label = "{}:{} MNI: {}".format(
            hemi_str, str(vertex_id).ljust(6),
            ', '.join('%5.1f' % m for m in mni))
        line = self.mpl_canvas.plot(
            time,
            self.act_data[hemi][vertex_id, :],
            label=label,
            lw=1.,
            color=color
        )
        return line

    def plot_time_line(self):
        if not hasattr(self, "mpl_canvas"):
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
            else:
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

    @safe_event
    def clean(self):
        # resolve the reference cycle
        self.clear_points()
        self.actions.clear()
        self.reps = None
        self._time_slider = None
        self._playback_speed_slider = None
        self.orientation_call.plotter = None
        self.orientation_call.brain = None
        self.orientation_call = None
        self.smoothing_call.plotter = None
        self.smoothing_call = None
        if self.time_call is not None:
            self.time_call.plotter = None
            self.time_call.brain = None
            self.time_call = None
            self.playback_speed_call.plotter = None
            self.playback_speed_call = None
        self.fmin_call.plotter = None
        self.fmin_call.brain = None
        self.fmin_call = None
        self.fmid_call.plotter = None
        self.fmid_call.brain = None
        self.fmid_call = None
        self.fmax_call.plotter = None
        self.fmax_call.brain = None
        self.fmax_call = None
        self.fscale_call.plotter = None
        self.fscale_call.brain = None
        self.fscale_call = None
        self.brain.time_viewer = None
        self.brain = None
        self.plotter = None
        self.main_menu = None
        self.window = None
        self.tool_bar = None
        self.status_bar = None
        self.interactor = None
        if hasattr(self, "mpl_canvas"):
            self.mpl_canvas.close()
            self.mpl_canvas.time_viewer = None
            self.mpl_canvas.canvas = None
            self.mpl_canvas = None
        self.time_actor = None
        self.picked_renderer = None
        self.act_data["lh"] = None
        self.act_data["rh"] = None
        self.act_data = None


class _LinkViewer(object):
    """Class to link multiple _TimeViewer objects."""

    def __init__(self, brains):
        self.brains = brains
        self.time_viewers = [brain.time_viewer for brain in brains]

        # link time sliders
        self.link_sliders(
            name="_time_slider",
            callback=self.set_time_point,
            event_type="always"
        )

        # link playback speed sliders
        self.link_sliders(
            name="_playback_speed_slider",
            callback=self.set_playback_speed,
            event_type="always"
        )

        # link toggle to start/pause playback
        for time_viewer in self.time_viewers:
            time_viewer.actions["play"].triggered.disconnect()
            time_viewer.actions["play"].triggered.connect(self.toggle_playback)

    def set_time_point(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.time_call(value, update_widget=True)

    def set_playback_speed(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.playback_speed_call(value, update_widget=True)

    def toggle_playback(self):
        master = self.time_viewers[0]  # select a master time_viewer
        value = master.time_call.slider_rep.GetValue()
        # synchronize starting points before playback
        self.set_time_point(value)
        for time_viewer in self.time_viewers:
            time_viewer.toggle_playback()

    def link_sliders(self, name, callback, event_type):
        from ..backends._pyvista import _update_slider_callback
        for time_viewer in self.time_viewers:
            slider = getattr(time_viewer, name, None)
            if slider is not None:
                _update_slider_callback(
                    slider=slider,
                    callback=callback,
                    event_type=event_type
                )


def _get_range(brain):
    val = np.abs(brain._current_act_data)
    return [np.min(val), np.max(val)]


def _normalize(point, shape):
    return (point[0] / shape[1], point[1] / shape[0])


@run_once
def _init_resources():
    from ...icons import resources
    resources.qInitResources()
