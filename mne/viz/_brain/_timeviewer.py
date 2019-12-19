# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        plotter = brain._renderer.plotter

        smoothing_slider = plotter.add_slider_widget(
            brain.set_data_smoothing,
            rng=[2, 30], title="smoothing",
            pointa=(0.95, -0.1),
            pointb=(0.95, 1.2)
        )

        def test(val):
            pass

        time_slider = plotter.add_slider_widget(
            test,
            rng=[0, 1], title="time",
            pointa=(0.85, -0.1),
            pointb=(0.85, 1.2)
        )

        _set_slider_style(smoothing_slider)
        _set_slider_style(time_slider)


def _set_slider_style(slider):
    slider_rep = slider.GetRepresentation()
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.04)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.02)
    slider_rep.GetSliderProperty().SetColor((0.5, 0.5, 0.5))
    slider_rep.GetTitleProperty().SetOrientation(-90)
