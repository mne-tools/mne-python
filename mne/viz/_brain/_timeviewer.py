# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class TextSliderHelper(object):
    """Class to set a text slider."""
    def __init__(self, slider=None, brain=None, orientation=None):
        self.slider = slider
        self.brain = brain
        self.orientation = orientation

    def __call__(self, idx):
        idx = int(idx)
        orientation = self.orientation[idx]
        if self.slider is not None:
            self.slider.GetRepresentation().SetTitleText(orientation)
            self.brain.show_view(orientation)


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        self.plotter = brain._renderer.plotter

        smoothing_slider = self.plotter.add_slider_widget(
            brain.set_data_smoothing,
            value=10,
            rng=[2, 30], title="smoothing",
            pointa=(0.82, 0.9),
            pointb=(0.98, 0.9)
        )

        max_time = len(brain._data['time']) - 1
        time_slider = self.plotter.add_slider_widget(
            brain.set_time_point,
            rng=[0, max_time], title="time",
            pointa=(0.85, 0.),
            pointb=(0.85, 1.),
            event_type='always'
        )

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
        set_orientation = TextSliderHelper(None, brain, orientation)
        orientation_slider = self.plotter.add_slider_widget(
            set_orientation,
            value=0,
            rng=[0, len(orientation) - 1],
            pointa=(0.95, 0.),
            pointb=(0.95, 1.),
            event_type='always'
        )
        set_orientation.slider = orientation_slider
        set_orientation(0)

        _set_slider_style(smoothing_slider)
        _set_slider_style(time_slider, vertical=True)
        _set_slider_style(orientation_slider, vertical=True)

        self.visibility = True
        self.plotter.add_key_event('y', self.toggle_interface)

    def toggle_interface(self):
        self.visibility = not self.visibility
        for slider in self.plotter.slider_widgets:
            if self.visibility:
                slider.On()
            else:
                slider.Off()


def _set_slider_style(slider, vertical=False):
    slider_rep = slider.GetRepresentation()
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.04)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.02)
    slider_rep.GetSliderProperty().SetColor((0.5, 0.5, 0.5))
    if vertical:
        slider_rep.GetTitleProperty().SetOrientation(-90)
