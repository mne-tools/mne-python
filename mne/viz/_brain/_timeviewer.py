# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


class _TimeViewer(object):
    """Class to interact with _Brain."""

    def __init__(self, brain):
        plotter = brain._renderer.plotter

        plotter.add_slider_widget(brain.set_data_smoothing, rng=[2, 30])
