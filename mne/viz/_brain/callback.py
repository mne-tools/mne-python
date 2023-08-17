# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class ShowView:
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
                row=self.data[idx]["row"],
                col=self.data[idx]["col"],
                hemi=self.data[idx]["hemi"],
            )
        if update_widget and self.widget is not None:
            self.widget.set_value(value)


class SmartCallBack:
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
