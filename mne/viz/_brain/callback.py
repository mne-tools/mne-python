# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import time


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
        self.slider_rep = None
        self._first_time = True

    def __call__(self, value):
        """Update the colorbar sliders."""
        if self._first_time:
            self._first_time = False
            return
        self.brain._update_fscale(value)
        for key in self.keys:
            if self.reps[key] is not None:
                self.reps[key].SetValue(self.brain._data[key])
        if self.slider_rep is not None:
            self.slider_rep.SetValue(1.0)
        self.plotter.update()


class BumpColorbarPoints(object):
    """Class that ensure constraints over the colorbar points."""

    def __init__(self, plotter=None, brain=None, name=None):
        self.plotter = plotter
        self.brain = brain
        self.name = name
        self.callback = {
            "fmin": lambda fmin: brain.update_lut(fmin=fmin),
            "fmid": lambda fmid: brain.update_lut(fmid=fmid),
            "fmax": lambda fmax: brain.update_lut(fmax=fmax),
        }
        self.keys = ('fmin', 'fmid', 'fmax')
        self.reps = {key: None for key in self.keys}
        self.last_update = time.time()
        self._first_time = True

    def __call__(self, value):
        """Update the colorbar sliders."""
        if self._first_time:
            self._first_time = False
            return
        vals = {key: self.brain._data[key] for key in self.keys}
        if self.name == "fmin" and self.reps["fmin"] is not None:
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.reps['fmax'].SetValue(value)
            if vals['fmid'] < value:
                vals['fmid'] = value
                self.reps['fmid'].SetValue(value)
            self.reps['fmin'].SetValue(value)
        elif self.name == "fmid" and self.reps['fmid'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.reps['fmin'].SetValue(value)
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.reps['fmax'].SetValue(value)
            self.reps['fmid'].SetValue(value)
        elif self.name == "fmax" and self.reps['fmax'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.reps['fmin'].SetValue(value)
            if vals['fmid'] > value:
                vals['fmid'] = value
                self.reps['fmid'].SetValue(value)
            self.reps['fmax'].SetValue(value)
        self.brain.update_lut(**vals)
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
