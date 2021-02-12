# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import time


class TimeCallBack(object):
    """Callback to update the time."""

    def __init__(self, brain=None, callback=None):
        self.brain = brain
        self.callback = callback
        self.widget = None
        self.time_label = None
        if self.brain is not None and callable(self.brain._data['time_label']):
            self.time_label = self.brain._data['time_label']

    def __call__(self, value, update_widget=False, time_as_index=True):
        """Update the time slider."""
        value = float(value)
        if not time_as_index:
            value = self.brain._to_time_index(value)
        self.brain.set_time_point(value)
        if self.callback is not None:
            self.callback()
        current_time = self.brain._current_time
        if self.widget is not None:
            if self.time_label is not None:
                current_time = self.time_label(current_time)
            if update_widget:
                self.widget.setValue(value)


class UpdateColorbarScale(object):
    """Class to update the values of the colorbar sliders."""

    def __init__(self, brain=None):
        self.brain = brain
        self.keys = ('fmin', 'fmid', 'fmax')
        self.widget = None
        self.widgets = {key: None for key in self.keys}

    def __call__(self, value, update_widget=False):
        """Update the colorbar sliders."""
        self.brain._update_fscale(value)
        for key in self.keys:
            if self.widgets[key] is not None:
                self.widgets[key].setValue(self.brain._data[key])
        if self.widget is not None:
            self.widget.setValue(1.0)


class BumpColorbarPoints(object):
    """Class that ensure constraints over the colorbar points."""

    def __init__(self, brain=None, name=None):
        self.brain = brain
        self.name = name
        self.callback = {
            "fmin": lambda fmin: brain.update_lut(fmin=fmin),
            "fmid": lambda fmid: brain.update_lut(fmid=fmid),
            "fmax": lambda fmax: brain.update_lut(fmax=fmax),
        }
        self.keys = ('fmin', 'fmid', 'fmax')
        self.widgets = {key: None for key in self.keys}
        self.last_update = time.time()

    def __call__(self, value):
        """Update the colorbar sliders."""
        vals = {key: self.brain._data[key] for key in self.keys}
        if self.name == "fmin" and self.widgets["fmin"] is not None:
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.widgets['fmax'].setValue(value)
            if vals['fmid'] < value:
                vals['fmid'] = value
                self.widgets['fmid'].setValue(value)
            self.widgets['fmin'].setValue(value)
        elif self.name == "fmid" and self.widgets['fmid'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.widgets['fmin'].setValue(value)
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.widgets['fmax'].setValue(value)
            self.widgets['fmid'].setValue(value)
        elif self.name == "fmax" and self.widgets['fmax'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.widgets['fmin'].setValue(value)
            if vals['fmid'] > value:
                vals['fmid'] = value
                self.widgets['fmid'].setValue(value)
            self.widgets['fmax'].setValue(value)
        self.brain.update_lut(**vals)
        if time.time() > self.last_update + 1. / 60.:
            self.callback[self.name](value)
            self.last_update = time.time()


class ShowView(object):
    """Class that selects the correct view."""

    def __init__(self, brain=None, data=None):
        self.brain = brain
        self.data = data
        self.widget = None

    def __call__(self, value, update_widget=False):
        """Update the view."""
        idx = self.brain.widgets["renderer"].value()
        if self.data[idx] is not None:
            self.brain.show_view(
                value,
                row=self.data[idx]['row'],
                col=self.data[idx]['col'],
                hemi=self.data[idx]['hemi'],
            )
        if update_widget and self.widget is not None:
            self.widget.setValue(value)


class SmartCallBack(object):
    """Class to manage smart slider.

    It stores it's own slider representation for efficiency
    and uses it when necessary.
    """

    def __init__(self, callback=None):
        self.callback = callback
        self.widget = None

    def __call__(self, value, update_widget=False):
        """Update the value."""
        self.callback(value)
        if update_widget and self.widget is not None:
            self.widget.setValue(value)
