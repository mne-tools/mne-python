# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class TextSlider(object):
    """Class to set a text slider."""

    def __init__(self, plotter=None, data=None,
                 callback=None, name=None):
        self.plotter = plotter
        self.data = data
        self.callback = callback
        self.name = name

    def __call__(self, idx):
        """Update the title of the slider."""
        idx = int(round(idx))
        data = self.data[idx]
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name == self.name:
                slider_rep = slider.GetRepresentation()
                slider_rep.SetTitleText(data)
                self.callback(data)


class IntSlider(object):
    """Class to set a integer slider."""

    def __init__(self, plotter=None, callback=None, name=None):
        self.plotter = plotter
        self.callback = callback
        self.name = name

    def __call__(self, idx):
        """Round the label of the slider."""
        idx = int(round(idx))
        for slider in self.plotter.slider_widgets:
            name = getattr(slider, "name", None)
            if name == self.name:
                slider_rep = slider.GetRepresentation()
                slider_rep.SetValue(idx)
                self.callback(idx)


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        self.plotter = brain._renderer.plotter

        # scalar bar
        scalar_bar = self.plotter.scalar_bar
        scalar_bar.SetOrientationToVertical()
        scalar_bar.SetHeight(0.47)
        scalar_bar.SetWidth(0.05)
        scalar_bar.SetPosition(0.095, 0.35)

        # smoothing slider
        default_smoothing_value = 7
        set_smoothing = IntSlider(
            plotter=self.plotter,
            callback=brain.set_data_smoothing,
            name="smoothing"
        )
        smoothing_slider = self.plotter.add_slider_widget(
            set_smoothing,
            value=default_smoothing_value,
            rng=[1, 15], title="smoothing",
            pointa=(0.82, 0.92),
            pointb=(0.98, 0.92)
        )
        smoothing_slider.name = 'smoothing'
        set_smoothing(default_smoothing_value)

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
        set_orientation = TextSlider(
            plotter=self.plotter,
            data=orientation,
            callback=brain.show_view,
            name="orientation"
        )
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

        # time label
        for hemi in brain._hemis:
            time_actor = brain._data[hemi + '_time_actor']
            if time_actor is not None:
                time_actor.SetPosition(0.5, 0.03)
                time_actor.GetTextProperty().SetJustificationToCentered()

        # time slider
        max_time = len(brain._data['time']) - 1
        time_slider = self.plotter.add_slider_widget(
            brain.set_time_point,
            value=brain._data['time_idx'],
            rng=[0, max_time],
            pointa=(0.25, 0.1),
            pointb=(0.75, 0.1),
            event_type='always'
        )

        # colormap slider
        fmin = brain._data["fmin"]
        fmin_slider = self.plotter.add_slider_widget(
            brain.update_fmin,
            value=fmin,
            rng=_get_range(fmin), title="fmin",
            pointa=(0.02, 0.27),
            pointb=(0.19, 0.27)
        )
        fmax = brain._data["fmax"]
        fmax_slider = self.plotter.add_slider_widget(
            brain.update_fmax,
            value=fmax,
            rng=_get_range(fmax), title="fmax",
            pointa=(0.02, 0.92),
            pointb=(0.19, 0.92)
        )

        # set the slider style
        _set_slider_style(smoothing_slider)
        _set_slider_style(orientation_slider, show_label=False)
        _set_slider_style(fmin_slider)
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
