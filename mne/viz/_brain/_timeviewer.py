# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import time
import warnings
import numpy as np


class IntSlider(object):
    """Class to set a integer slider."""

    def __init__(self, plotter=None, callback=None, name=None):
        self.plotter = plotter
        self.callback = callback
        self.name = name

    def __call__(self, value):
        """Round the label of the slider."""
        idx = int(round(value))
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name == self.name:
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(idx)
                self.callback(idx)


class UpdateColorbarScale(object):
    """Class to update the values of the colorbar sliders."""

    def __init__(self, plotter=None, brain=None):
        self.plotter = plotter
        self.brain = brain

    def __call__(self, value):
        """Update the colorbar sliders."""
        self.brain.update_fscale(value)
        fmin = self.brain._data['fmin'] * value
        fmid = self.brain._data['fmid'] * value
        fmax = self.brain._data['fmax'] * value
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


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        self.brain = brain
        self.plotter = brain._renderer.plotter

        # scalar bar
        if brain._colorbar_added:
            scalar_bar = self.plotter.scalar_bar
            scalar_bar.SetOrientationToVertical()
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetWidth(0.05)
            scalar_bar.SetPosition(0.02, 0.35)

        # smoothing slider
        default_smoothing_value = 7
        self.set_smoothing = IntSlider(
            plotter=self.plotter,
            callback=brain.set_data_smoothing,
            name="smoothing"
        )
        smoothing_slider = self.plotter.add_slider_widget(
            self.set_smoothing,
            value=default_smoothing_value,
            rng=[0, 15], title="smoothing",
            pointa=(0.82, 0.90),
            pointb=(0.98, 0.90)
        )
        smoothing_slider.name = 'smoothing'
        self.set_smoothing(default_smoothing_value)

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
        orientation_slider = self.plotter.add_text_slider_widget(
            brain.show_view,
            value=0,
            data=orientation,
            pointa=(0.82, 0.74),
            pointb=(0.98, 0.74),
            event_type='always'
        )

        # time label
        for hemi in brain._hemis:
            self.time_actor = brain._data.get(hemi + '_time_actor')
            if self.time_actor is not None:
                self.time_actor.SetPosition(0.5, 0.03)
                self.time_actor.GetTextProperty().SetJustificationToCentered()

        # time slider
        max_time = len(brain._data['time']) - 1
        time_slider = self.plotter.add_slider_widget(
            brain.set_time_point,
            value=brain._data['time_idx'],
            rng=[0, max_time],
            pointa=(0.23, 0.1),
            pointb=(0.77, 0.1),
            event_type='always'
        )
        time_slider.name = "time_slider"

        # playback speed
        default_playback_speed = 1
        self.set_playback_speed = IntSlider(
            plotter=self.plotter,
            callback=self.set_playback_speed,
            name="playback_speed"
        )
        playback_speed_slider = self.plotter.add_slider_widget(
            self.set_playback_speed,
            value=default_playback_speed,
            rng=[1, 100], title="playback speed",
            pointa=(0.02, 0.1),
            pointb=(0.18, 0.1)
        )
        playback_speed_slider.name = "playback_speed"

        # colormap slider
        scaling_limits = [0.2, 2.0]
        fmin = brain._data["fmin"]
        self.update_fmin = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmin"
        )
        fmin_slider = self.plotter.add_slider_widget(
            self.update_fmin,
            value=fmin,
            rng=_get_range(brain), title="fmin",
            pointa=(0.82, 0.26),
            pointb=(0.98, 0.26),
            event_type="always",
        )
        fmin_slider.name = "fmin"
        fmid = brain._data["fmid"]
        self.update_fmid = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmid",
        )
        fmid_slider = self.plotter.add_slider_widget(
            self.update_fmid,
            value=fmid,
            rng=_get_range(brain), title="fmid",
            pointa=(0.82, 0.42),
            pointb=(0.98, 0.42),
            event_type="always",
        )
        fmid_slider.name = "fmid"
        fmax = brain._data["fmax"]
        self.update_fmax = BumpColorbarPoints(
            plotter=self.plotter,
            brain=brain,
            name="fmax",
        )
        fmax_slider = self.plotter.add_slider_widget(
            self.update_fmax,
            value=fmax,
            rng=_get_range(brain), title="fmax",
            pointa=(0.82, 0.58),
            pointb=(0.98, 0.58),
            event_type="always",
        )
        fmax_slider.name = "fmax"
        update_fscale = UpdateColorbarScale(
            plotter=self.plotter,
            brain=brain,
        )
        fscale_slider = self.plotter.add_slider_widget(
            update_fscale,
            value=1.0,
            rng=scaling_limits, title="fscale",
            pointa=(0.82, 0.10),
            pointb=(0.98, 0.10)
        )

        # add toggle to start/stop playback
        self.playback = False
        self.playback_speed = 1
        self.time_elapsed = 0
        self.refresh_rate = 16
        self.plotter.add_callback(self.play, self.refresh_rate)
        self.plotter.add_callback(self.perform_maintenance)
        self.button_size = 40
        self.font_size = 14
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            playback_button = self.plotter.add_checkbox_button_widget(
                self.toggle_playback,
                value=False,
                size=self.button_size,
                position=(0, 0)
            )
        playback_button.name = "toggle_playback"
        self.playback_actor = self.plotter.add_text(
            text="Start",
            font_size=self.font_size,
            position=(0, 0)
        )

        # add toggle to show/hide interface
        self.visibility = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            interface_button = self.plotter.add_checkbox_button_widget(
                self.toggle_interface,
                value=True,
                size=self.button_size,
                position=(0, 0)
            )
        interface_button.name = "toggle_interface"
        self.interface_actor = self.plotter.add_text(
            text="Hide",
            font_size=self.font_size,
            position=(0, 0)
        )

        # set the slider style
        _set_slider_style(smoothing_slider)
        _set_slider_style(orientation_slider, show_label=False)
        _set_slider_style(fmin_slider)
        _set_slider_style(fmid_slider)
        _set_slider_style(fmax_slider)
        _set_slider_style(fscale_slider)
        _set_slider_style(playback_speed_slider)
        _set_slider_style(time_slider, show_label=False)

        # set the text style
        _set_text_style(self.time_actor)
        _set_text_style(self.playback_actor)
        _set_text_style(self.interface_actor)

        self.perform_maintenance()

    def toggle_interface(self, state):
        self.visibility = state
        for slider in self.plotter.slider_widgets:
            if self.visibility:
                slider.On()
            else:
                slider.Off()
        if self.visibility:
            self.interface_actor.SetInput("Hide")
        else:
            self.interface_actor.SetInput("Show")

    def toggle_playback(self, state):
        self.playback = state
        self.time_elapsed = 0
        if self.playback:
            self.playback_actor.SetInput("Stop")
        else:
            self.playback_actor.SetInput("Start")

    def set_playback_speed(self, speed):
        self.playback_speed = speed

    def play(self):
        from scipy.interpolate import interp1d
        if self.playback:
            self.time_elapsed += self.refresh_rate
            if self.time_elapsed >= self.playback_speed * 10:
                time_data = self.brain._data['time']
                time_idx = self.brain._data['time_idx']
                times = np.arange(self.brain._n_times)
                ifunc = interp1d(times, time_data)
                time_point = ifunc(time_idx) + 1. / self.playback_speed
                idx = np.argmin(np.abs(time_data - time_point))

                max_time = len(self.brain._data['time'])
                if time_idx < max_time:
                    self.brain.set_time_point(idx)
                    for slider in self.plotter.slider_widgets:
                        name = getattr(slider, "name", None)
                        if name == "time_slider":
                            slider_rep = slider.GetRepresentation()
                            slider_rep.SetValue(idx)
                else:
                    self.playback = False
                self.time_elapsed = 0

    def place_widget(self, position):
        if hasattr(self.plotter, 'ren_win'):
            window_size = self.plotter.ren_win.GetSize()
            position = (
                position[0] * window_size[0],
                position[1] * window_size[1]
            )
        return position

    def set_bounds(self, position):
        bounds = [
            position[0], position[0] + self.button_size,
            position[1], position[1] + self.button_size,
            0., 0.
        ]
        return bounds

    def perform_maintenance(self):
        for button in self.plotter.button_widgets:
            name = getattr(button, "name", None)
            if name == "toggle_playback":
                button_rep = button.GetRepresentation()
                position = self.place_widget((0.02, 0.17))
                bounds = self.set_bounds(position)
                button_rep.PlaceWidget(bounds)
            elif name == "toggle_interface":
                button_rep = button.GetRepresentation()
                position = self.place_widget((0.02, 0.27))
                bounds = self.set_bounds(position)
                button_rep.PlaceWidget(bounds)

        self.playback_actor.SetPosition(self.place_widget((0.06, 0.17)))
        self.interface_actor.SetPosition(self.place_widget((0.06, 0.27)))


def _set_slider_style(slider, show_label=True):
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


def _set_text_style(text_actor):
    if text_actor is not None:
        prop = text_actor.GetTextProperty()
        prop.BoldOn()


def _get_range(brain):
    val = np.abs(brain._data['array'])
    return [np.min(val), np.max(val)]
