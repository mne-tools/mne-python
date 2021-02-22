# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import time


class Widget(object):
    """Helper class to interface widgets."""

    def __init__(self, widget, notebook=False):
        self.widget = widget
        self.notebook = notebook

    def set_value(self, value):
        """Set the widget value."""
        if self.notebook:
            self.widget.value = value
        else:
            if hasattr(self.widget, "setValue"):
                self.widget.setValue(value)
            elif hasattr(self.widget, "setCurrentText"):
                self.widget.setCurrentText(value)

    def get_value(self):
        """Get the widget value."""
        if self.notebook:
            return self.widget.value
        else:
            if hasattr(self.widget, "value"):
                return self.widget.value()
            elif hasattr(self.widget, "currentText"):
                return self.widget.currentText()


class TimeCallBack(object):
    """Callback to update the time."""

    def __init__(self, brain=None, callback=None):
        self.brain = brain
        self.callback = callback
        self.widget = None
        if self.brain is not None and callable(self.brain._data['time_label']):
            self.time_label = self.brain._data['time_label']
        else:
            self.time_label = None

    def __call__(self, value, update_widget=False, time_as_index=True):
        """Update the time slider."""
        if not time_as_index:
            value = self.brain._to_time_index(value)
        self.brain.set_time_point(value)
        if self.callback is not None:
            self.callback()
        current_time = self.brain._current_time
        if self.time_label is not None:
            current_time = self.time_label(current_time)
        if self.widget is not None and update_widget:
            self.widget.set_value(int(value))


class UpdateColorbarScale(object):
    """Class to update the values of the colorbar sliders."""

    def __init__(self, brain=None):
        self.brain = brain
        self.widget = None
        self.widgets = {key: None for key in self.brain.keys}

    def __call__(self, value):
        """Update the colorbar sliders."""
        self.brain._update_fscale(value)
        for key in self.brain.keys:
            if self.widgets[key] is not None:
                self.widgets[key].set_value(self.brain._data[key])
        if self.widget is not None:
            self.widget.set_value(1.0)


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
        self.widgets = {key: None for key in self.brain.keys}
        self.last_update = time.time()

    def __call__(self, value):
        """Update the colorbar sliders."""
        vals = {key: self.brain._data[key] for key in self.brain.keys}
        if self.name == "fmin" and self.widgets["fmin"] is not None:
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.widgets['fmax'].set_value(value)
            if vals['fmid'] < value:
                vals['fmid'] = value
                self.widgets['fmid'].set_value(value)
            self.widgets['fmin'].set_value(value)
        elif self.name == "fmid" and self.widgets['fmid'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.widgets['fmin'].set_value(value)
            if vals['fmax'] < value:
                vals['fmax'] = value
                self.widgets['fmax'].set_value(value)
            self.widgets['fmid'].set_value(value)
        elif self.name == "fmax" and self.widgets['fmax'] is not None:
            if vals['fmin'] > value:
                vals['fmin'] = value
                self.widgets['fmin'].set_value(value)
            if vals['fmid'] > value:
                vals['fmid'] = value
                self.widgets['fmid'].set_value(value)
            self.widgets['fmax'].set_value(value)
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
        if "renderer" in self.brain.widgets:
            idx = self.brain.widgets["renderer"].get_value()
        else:
            idx = 0
        idx = int(idx)
        if self.data[idx] is not None:
            self.brain.show_view(
                value,
                row=self.data[idx]['row'],
                col=self.data[idx]['col'],
                hemi=self.data[idx]['hemi'],
            )
        if update_widget and self.widget is not None:
            self.widget.set_value(value)


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
        if self.widget is not None and update_widget:
            self.widget.set_value(value)
