# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class TextSliderHelper(object):
    """Class to set a text slider."""

    def __init__(self, plotter=None, brain=None, orientation=None):
        self.plotter = plotter
        self.brain = brain
        self.orientation = orientation

    def __call__(self, idx):
        """Update the title of the slider."""
        idx = int(idx)
        orientation = self.orientation[idx]
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name == "orientation":
                slider_rep = slider.GetRepresentation()
                slider_rep.SetTitleText(orientation)
                self.brain.show_view(orientation)


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        self.plotter = brain._renderer.plotter

        # smoothing slider
        smoothing_slider = self.plotter.add_slider_widget(
            brain.set_data_smoothing,
            value=7,
            rng=[1, 15], title="smoothing",
            pointa=(0.82, 0.92),
            pointb=(0.98, 0.92)
        )

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
        set_orientation = TextSliderHelper(self.plotter, brain, orientation)
        orientation_slider = self.plotter.add_slider_widget(
            set_orientation,
            value=0,
            rng=[0, len(orientation) - 1],
            pointa=(0.82, 0.78),
            pointb=(0.98, 0.78),
            event_type='always'
        )
        orientation_slider.name = "orientation"
        set_orientation(0)

        # time slider
        max_time = len(brain._data['time']) - 1
        time_slider = self.plotter.add_slider_widget(
            brain.set_time_point,
            value=brain._data['time_idx'],
            rng=[0, max_time],
            pointa=(0.15, 0.16),
            pointb=(0.85, 0.16),
            event_type='always'
        )

        # colormap slider
        fmin = brain._data["fmin"]
        fmin_slider = self.plotter.add_slider_widget(
            brain.update_fmin,
            value=fmin,
            rng=_get_range(fmin), title="fmin",
            pointa=(0.02, 0.92),
            pointb=(0.19, 0.92)
        )
        fmid = brain._data["fmid"]
        fmid_slider = self.plotter.add_slider_widget(
            brain.update_fmid,
            value=fmid,
            rng=_get_range(fmid), title="fmid",
            pointa=(0.02, 0.78),
            pointb=(0.19, 0.78)
        )
        fmax = brain._data["fmax"]
        fmax_slider = self.plotter.add_slider_widget(
            brain.update_fmax,
            value=fmax,
            rng=_get_range(fmax), title="fmax",
            pointa=(0.02, 0.64),
            pointb=(0.19, 0.64)
        )

        # set the slider style
        _set_slider_style(smoothing_slider)
        _set_slider_style(orientation_slider, show_label=False)
        _set_slider_style(fmin_slider)
        _set_slider_style(fmid_slider)
        _set_slider_style(fmax_slider)
        _set_slider_style(time_slider, show_label=False)

        # add toggle to show/hide interface
        self.visibility = True
        self.plotter.add_key_event('y', self.toggle_interface)

    def toggle_interface(self):
        self.visibility = not self.visibility
        for slider in self.plotter.slider_widgets:
            if self.visibility:
                slider.On()
            else:
                slider.Off()


def _set_slider_style(slider, show_label=True):
    slider_rep = slider.GetRepresentation()
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.04)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.02)
    slider_rep.GetSliderProperty().SetColor((0.5, 0.5, 0.5))
    if not show_label:
        slider_rep.ShowSliderLabelOff()


def _get_range(val, percentage=0.5):
    mid = abs(val) * percentage
    return [val - mid, val + mid]
