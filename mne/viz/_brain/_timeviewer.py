# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from itertools import cycle
import time
import numpy as np
from ..utils import _show_help, _get_color_list, tight_layout
from ...source_space import vertex_to_mni


class MplCanvas(object):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent, width, height, dpi):
        from PyQt5 import QtWidgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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

    def plot(self, x, y, label, **kwargs):
        """Plot a curve."""
        line, = self.axes.plot(
            x, y, label=label, **kwargs)
        self.update_plot()
        return line

    def update_plot(self):
        """Update the plot."""
        self.axes.legend(prop={'family': 'monospace', 'size': 'small'},
                         framealpha=0.5, handlelength=1.)
        self.canvas.draw()

    def show(self):
        """Show the canvas."""
        self.canvas.show()

    def close(self):
        """Close the canvas."""
        self.canvas.close()


class IntSlider(object):
    """Class to set a integer slider."""

    def __init__(self, plotter=None, callback=None, name=None):
        self.plotter = plotter
        self.callback = callback
        self.name = name
        self.slider_rep = None

    def __call__(self, value):
        """Round the label of the slider."""
        idx = int(round(value))
        if self.slider_rep is None:
            for slider in self.plotter.slider_widgets:
                name = getattr(slider, "name", None)
                if name == self.name:
                    self.slider_rep = slider.GetRepresentation()
        if self.slider_rep is not None:
            self.slider_rep.SetValue(idx)
        self.callback(idx)


class TimeSlider(object):
    """Class to update the time slider."""

    def __init__(self, plotter=None, brain=None):
        self.plotter = plotter
        self.brain = brain
        self.slider_rep = None
        if brain is None:
            self.time_label = None
        else:
            if callable(self.brain._data['time_label']):
                self.time_label = self.brain._data['time_label']

    def __call__(self, value, update_widget=False):
        """Update the time slider."""
        self.brain.set_time_point(value)
        current_time = self.brain._current_time
        if self.slider_rep is None:
            for slider in self.plotter.slider_widgets:
                name = getattr(slider, "name", None)
                if name == "time":
                    self.slider_rep = slider.GetRepresentation()
        if self.slider_rep is not None:
            if update_widget:
                self.slider_rep.SetValue(value)
            if self.time_label is not None:
                current_time = self.time_label(current_time)
                self.slider_rep.SetTitleText(current_time)


class UpdateColorbarScale(object):
    """Class to update the values of the colorbar sliders."""

    def __init__(self, plotter=None, brain=None):
        self.plotter = plotter
        self.brain = brain
        self.slider_rep = None

    def __call__(self, value):
        """Update the colorbar sliders."""
        self.brain.update_fscale(value)
        fmin = self.brain._data['fmin']
        fmid = self.brain._data['fmid']
        fmax = self.brain._data['fmax']
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name == "fmin":
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(fmin)
            elif name == "fmid":
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(fmid)
            elif name == "fmax":
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(fmax)
            elif name == "fscale":
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(1.0)


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
        self.last_update = time.time()

    def __call__(self, value):
        """Update the colorbar sliders."""
        keys = ('fmin', 'fmid', 'fmax')
        vals = {key: self.brain._data[key] for key in keys}
        reps = {key: None for key in keys}
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name is not None:
                reps[name] = slider.GetRepresentation()
        if self.name == "fmin" and reps["fmin"] is not None:
            if vals['fmax'] < value:
                self.brain.update_fmax(value)
                reps['fmax'].SetValue(value)
            if vals['fmid'] < value:
                self.brain.update_fmid(value)
                reps['fmid'].SetValue(value)
            reps['fmin'].SetValue(value)
        elif self.name == "fmid" and reps['fmid'] is not None:
            if vals['fmin'] > value:
                self.brain.update_fmin(value)
                reps['fmin'].SetValue(value)
            if vals['fmax'] < value:
                self.brain.update_fmax(value)
                reps['fmax'].SetValue(value)
            reps['fmid'].SetValue(value)
        elif self.name == "fmax" and reps['fmax'] is not None:
            if vals['fmin'] > value:
                self.brain.update_fmin(value)
                reps['fmin'].SetValue(value)
            if vals['fmid'] > value:
                self.brain.update_fmid(value)
                reps['fmid'].SetValue(value)
            reps['fmax'].SetValue(value)
        if time.time() > self.last_update + 1. / 60.:
            self.callback[self.name](value)
            self.last_update = time.time()


class ShowView(object):
    """Class that selects the correct view."""

    def __init__(self, plotter=None, brain=None, orientation=None,
                 row=None, col=None, hemi=None, name=None):
        self.plotter = plotter
        self.brain = brain
        self.orientation = orientation
        self.short_orientation = [s[:3] for s in orientation]
        self.row = row
        self.col = col
        self.hemi = hemi
        self.name = name
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
            if self.slider_rep is None:
                for slider in self.plotter.slider_widgets:
                    name = getattr(slider, "name", None)
                    if name == self.name:
                        self.slider_rep = slider.GetRepresentation()
            if self.slider_rep is not None:
                self.slider_rep.SetValue(idx)
                self.slider_rep.SetTitleText(self.orientation[idx])


class SmartSlider(object):
    """Class to manage smart slider.

    It stores it's own slider representation for efficiency
    and uses it when necessary.
    """

    def __init__(self, plotter=None, callback=None, name=None):
        self.plotter = plotter
        self.callback = callback
        self.name = name
        self.slider_rep = None

    def __call__(self, value, update_widget=False):
        """Update the value."""
        self.callback(value)
        if update_widget:
            if self.slider_rep is None:
                for slider in self.plotter.slider_widgets:
                    name = getattr(slider, "name", None)
                    if name == self.name:
                        self.slider_rep = slider.GetRepresentation()
            if self.slider_rep is not None:
                self.slider_rep.SetValue(value)


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain, show_traces=False):
        self.brain = brain
        self.brain.time_viewer = self
        self.plotter = brain._renderer.plotter
        self.interactor = self.plotter
        self.interactor.keyPressEvent = self.keyPressEvent

        # orientation slider
        orientation = [
            'lateral',
            'medial',
            'rostral',
            'caudal',
            'dorsal',
            'ventral',
            'frontal',
            'parietal'
        ]

        # default: put orientation slider on the first view
        if self.brain._hemi == 'split':
            self.plotter.subplot(0, 0)

        for hemi in self.brain._hemis:
            if self.brain._hemi == 'split':
                ci = 0 if hemi == 'lh' else 1
            else:
                ci = 0
            for ri, view in enumerate(self.brain._views):
                self.plotter.subplot(ri, ci)
                name = "orientation_" + str(ri) + "_" + str(ci)
                self.orientation_call = ShowView(
                    plotter=self.plotter,
                    brain=self.brain,
                    orientation=orientation,
                    hemi=hemi,
                    row=ri,
                    col=ci,
                    name=name
                )
                orientation_slider = self.plotter.add_text_slider_widget(
                    self.orientation_call,
                    value=0,
                    data=orientation,
                    pointa=(0.82, 0.74),
                    pointb=(0.98, 0.74),
                    event_type='always'
                )
                orientation_slider.name = name
                self.set_slider_style(orientation_slider, show_label=False)
                self.orientation_call(view, update_widget=True)

        # necessary because show_view modified subplot
        if self.brain._hemi == 'split':
            self.plotter.subplot(0, 0)

        # scalar bar
        if brain._colorbar_added:
            scalar_bar = self.plotter.scalar_bar
            scalar_bar.SetOrientationToVertical()
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetWidth(0.05)
            scalar_bar.SetPosition(0.02, 0.2)

        # smoothing slider
        default_smoothing_value = 7
        self.smoothing_call = IntSlider(
            plotter=self.plotter,
            callback=brain.set_data_smoothing,
            name="smoothing"
        )
        smoothing_slider = self.plotter.add_slider_widget(
            self.smoothing_call,
            value=default_smoothing_value,
            rng=[0, 15], title="smoothing",
            pointa=(0.82, 0.90),
            pointb=(0.98, 0.90)
        )
        smoothing_slider.name = 'smoothing'
        self.smoothing_call(default_smoothing_value)

        # time label
        self.time_actor = brain._data.get('time_actor')
        if self.time_actor is not None:
            self.time_actor.SetPosition(0.5, 0.03)
            self.time_actor.GetTextProperty().SetJustificationToCentered()
            self.time_actor.GetTextProperty().BoldOn()
            self.time_actor.VisibilityOff()

        # time slider
        max_time = len(brain._data['time']) - 1
        self.time_call = TimeSlider(
            plotter=self.plotter,
            brain=self.brain
        )
        time_slider = self.plotter.add_slider_widget(
            self.time_call,
            value=self.brain._data['time_idx'],
            rng=[0, max_time],
            pointa=(0.23, 0.1),
            pointb=(0.77, 0.1),
            event_type='always'
        )
        time_slider.GetRepresentation().SetLabelFormat('idx=%0.1f')

        time_slider.name = "time"
        # set the default value
        self.time_call(value=brain._data['time_idx'])

        # playback speed
        default_playback_speed = 0.05
        self.playback_speed_call = SmartSlider(
            plotter=self.plotter,
            callback=self.set_playback_speed,
            name="playback_speed"
        )
        playback_speed_slider = self.plotter.add_slider_widget(
            self.playback_speed_call,
            value=default_playback_speed,
            rng=[0.01, 1], title="speed",
            pointa=(0.02, 0.1),
            pointb=(0.18, 0.1),
            event_type='always'
        )
        playback_speed_slider.name = "playback_speed"

        # colormap slider
        scaling_limits = [0.2, 2.0]
        pointa = np.array((0.82, 0.26))
        pointb = np.array((0.98, 0.26))
        shift = np.array([0, 0.08])
        fmin = brain._data["fmin"]
        self.fmin_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmin"
        )
        fmin_slider = self.plotter.add_slider_widget(
            self.fmin_call,
            value=fmin,
            rng=_get_range(brain), title="clim",
            pointa=pointa,
            pointb=pointb,
            event_type="always",
        )
        fmin_slider.name = "fmin"
        self.fmin_slider_rep = fmin_slider.GetRepresentation()
        fmid = brain._data["fmid"]
        self.fmid_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmid",
        )
        fmid_slider = self.plotter.add_slider_widget(
            self.fmid_call,
            value=fmid,
            rng=_get_range(brain), title="",
            pointa=pointa + shift,
            pointb=pointb + shift,
            event_type="always",
        )
        fmid_slider.name = "fmid"
        self.fmid_slider_rep = fmid_slider.GetRepresentation()
        fmax = brain._data["fmax"]
        self.fmax_call = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmax",
        )
        fmax_slider = self.plotter.add_slider_widget(
            self.fmax_call,
            value=fmax,
            rng=_get_range(brain), title="",
            pointa=pointa + 2 * shift,
            pointb=pointb + 2 * shift,
            event_type="always",
        )
        fmax_slider.name = "fmax"
        self.fmax_slider_rep = fmax_slider.GetRepresentation()
        self.fscale_call = UpdateColorbarScale(
            plotter=self.plotter,
            brain=brain,
        )
        fscale_slider = self.plotter.add_slider_widget(
            self.fscale_call,
            value=1.0,
            rng=scaling_limits, title="fscale",
            pointa=(0.82, 0.10),
            pointb=(0.98, 0.10)
        )
        fscale_slider.name = "fscale"

        # add toggle to start/pause playback
        self.playback = False
        self.playback_speed = default_playback_speed
        self.refresh_rate_ms = max(int(round(1000. / 60.)), 1)
        self.plotter.add_callback(self.play, self.refresh_rate_ms)

        # add toggle to show/hide interface
        self.visibility = True

        # set the slider style
        self.set_slider_style(smoothing_slider)
        self.set_slider_style(fmin_slider)
        self.set_slider_style(fmid_slider)
        self.set_slider_style(fmax_slider)
        self.set_slider_style(fscale_slider)
        self.set_slider_style(playback_speed_slider)
        self.set_slider_style(time_slider)

        # Point Picking and MplCanvas plotting
        if isinstance(show_traces, str) and show_traces == "separate":
            show_traces = True
            self.separate_canvas = True
        else:
            self.separate_canvas = False
        if isinstance(show_traces, bool) and show_traces:
            self.act_data = {'lh': None, 'rh': None}
            self.color_cycle = None
            self.picked_points = {'lh': list(), 'rh': list()}
            self._mouse_no_mvt = -1
            self.enable_point_picking()

        # remove default picking menu
        main_menu = self.plotter.main_menu
        to_remove = list()
        for action in main_menu.actions():
            if action.text() == "Tools":
                to_remove.append(action)
        for action in to_remove:
            main_menu.removeAction(action)

        # setup key bindings
        self.key_bindings = {
            '?': self.help,
            'i': self.toggle_interface,
            's': self.apply_auto_scaling,
            'r': self.restore_user_scaling,
            ' ': self.toggle_playback,
        }
        menu = self.plotter.main_menu.addMenu('Help')
        menu.addAction('Show MNE key bindings\t?', self.help)

    def keyPressEvent(self, event):
        callback = self.key_bindings.get(event.text())
        if callback is not None:
            callback()

    def toggle_interface(self):
        self.visibility = not self.visibility

        # manage sliders
        for slider in self.plotter.slider_widgets:
            slider_rep = slider.GetRepresentation()
            if self.visibility:
                slider_rep.VisibilityOn()
            else:
                slider_rep.VisibilityOff()

        # manage time label
        time_label = self.brain._data['time_label']
        if callable(time_label) and self.time_actor is not None:
            if self.visibility:
                self.time_actor.VisibilityOff()
            else:
                self.time_actor.SetInput(time_label(self.brain._current_time))
                self.time_actor.VisibilityOn()

    def apply_auto_scaling(self):
        self.brain.update_auto_scaling()
        self.fmin_slider_rep.SetValue(self.brain._data['fmin'])
        self.fmid_slider_rep.SetValue(self.brain._data['fmid'])
        self.fmax_slider_rep.SetValue(self.brain._data['fmax'])

    def restore_user_scaling(self):
        self.brain.update_auto_scaling(restore=True)
        self.fmin_slider_rep.SetValue(self.brain._data['fmin'])
        self.fmid_slider_rep.SetValue(self.brain._data['fmid'])
        self.fmax_slider_rep.SetValue(self.brain._data['fmax'])

    def toggle_playback(self):
        self.playback = not self.playback
        if self.playback:
            time_data = self.brain._data['time']
            max_time = np.max(time_data)
            if self.brain._current_time == max_time:  # start over
                self.brain.set_time_point(np.min(time_data))
            self._last_tick = time.time()

    def set_playback_speed(self, speed):
        self.playback_speed = speed

    def play(self):
        if self.playback:
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
                self.playback = False
            self.plotter.update()  # critical for smooth animation

    def set_slider_style(self, slider, show_label=True):
        if slider is not None:
            slider_rep = slider.GetRepresentation()
            slider_rep.SetSliderLength(0.02)
            slider_rep.SetSliderWidth(0.04)
            slider_rep.SetTubeWidth(0.005)
            slider_rep.SetEndCapLength(0.01)
            slider_rep.SetEndCapWidth(0.02)
            slider_rep.GetSliderProperty().SetColor((0.5, 0.5, 0.5))
            if not show_label:
                slider_rep.ShowSliderLabelOff()

    def enable_point_picking(self):
        from ..backends._pyvista import _update_picking_callback
        # use a matplotlib canvas
        self.color_cycle = cycle(_get_color_list())
        win = self.plotter.app_window
        dpi = win.windowHandle().screen().logicalDotsPerInch()
        w, h = win.geometry().width() / dpi, win.geometry().height() / dpi
        h /= 3  # one third of the window
        if self.separate_canvas:
            parent = None
        else:
            parent = win
        self.mpl_canvas = MplCanvas(parent, w, h, dpi)
        xlim = [np.min(self.brain._data['time']),
                np.max(self.brain._data['time'])]
        self.mpl_canvas.axes.set(xlim=xlim)
        vlayout = self.plotter.frame.layout()
        if self.separate_canvas:
            self.plotter.app_window.signal_close.connect(self.mpl_canvas.close)
            self.mpl_canvas.show()
        else:
            vlayout.addWidget(self.mpl_canvas.canvas)
            vlayout.setStretch(0, 2)
            vlayout.setStretch(1, 1)

        # get brain data
        for idx, hemi in enumerate(['lh', 'rh']):
            hemi_data = self.brain._data.get(hemi)
            if hemi_data is not None:
                self.act_data[hemi] = hemi_data['array']
                smooth_mat = hemi_data['smooth_mat']
                if smooth_mat is not None:
                    self.act_data[hemi] = smooth_mat.dot(self.act_data[hemi])

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

        _update_picking_callback(
            self.plotter,
            self.on_mouse_move,
            self.on_button_press,
            self.on_button_release,
            self.on_pick
        )

    def on_mouse_move(self, vtk_picker, event):
        if self._mouse_no_mvt:
            self._mouse_no_mvt -= 1

    def on_button_press(self, vtk_picker, event):
        self._mouse_no_mvt = 2

    def on_button_release(self, vtk_picker, event):
        if self._mouse_no_mvt:
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
            cell = mesh.faces[cell_id][1:]
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
        center = mesh.GetPoints().GetPoint(vertex_id)

        # from the picked renderer to the subplot coords
        rindex = self.plotter.renderers.index(self.picked_renderer)
        row, col = self.plotter.index_to_loc(rindex)

        actors = list()
        spheres = list()
        for ri, view in enumerate(self.brain._views):
            self.plotter.subplot(ri, col)
            actor, sphere = self.brain._renderer.sphere(
                center=np.array(center),
                color=color,
                scale=1.0,
                radius=4.0
            )
            actors.append(actor)
            spheres.append(sphere)

        # add metadata for picking
        for sphere in spheres:
            sphere._is_point = True
            sphere._hemi = hemi
            sphere._line = line
            sphere._actors = actors
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
        self.plotter.remove_actor(mesh._actors)

    def plot_time_course(self, hemi, vertex_id, color):
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

    def help(self):
        pairs = [
            ('?', 'Display help window'),
            ('i', 'Toggle interface'),
            ('s', 'Apply auto-scaling'),
            ('r', 'Restore original clim'),
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


class _LinkViewer(object):
    """Class to link multiple _TimeViewer objects."""

    def __init__(self, brains):
        self.brains = brains
        self.time_viewers = [brain.time_viewer for brain in brains]

        # link time sliders
        self.link_sliders(
            name="time",
            callback=self.set_time_point,
            event_type="always"
        )

        # link playback speed sliders
        self.link_sliders(
            name="playback_speed",
            callback=self.set_playback_speed,
            event_type="always"
        )

        # link toggle to start/pause playback
        for time_viewer in self.time_viewers:
            time_viewer.key_bindings[' '] = self.toggle_playback

    def set_time_point(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.time_call(value, update_widget=True)

    def set_playback_speed(self, value):
        for time_viewer in self.time_viewers:
            time_viewer.playback_speed_call(value, update_widget=True)

    def toggle_playback(self):
        for time_viewer in self.time_viewers:
            time_viewer.toggle_playback()

    def link_sliders(self, name, callback, event_type):
        from ..backends._pyvista import _update_slider_callback
        for time_viewer in self.time_viewers:
            plotter = time_viewer.plotter
            for slider in plotter.slider_widgets:
                slider_name = getattr(slider, "name", None)
                if slider_name == name:
                    _update_slider_callback(
                        slider=slider,
                        callback=callback,
                        event_type=event_type
                    )


def _get_range(brain):
    val = np.abs(brain._data['array'])
    return [np.min(val), np.max(val)]


def _normalize(point, shape):
    return (point[0] / shape[1], point[1] / shape[0])
