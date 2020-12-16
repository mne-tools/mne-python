# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import matplotlib.pyplot as plt
from contextlib import contextmanager
from ...fixes import nullcontext
from ._pyvista import _Renderer as _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer):
    def __init__(self, *args, **kwargs):
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic('matplotlib widget')
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def show(self):
        self.figure.display = _NotebookInteractor(self)
        return self.scene()


class _NotebookInteractor(object):
    def __init__(self, renderer):
        from IPython import display
        from ipywidgets import HBox, VBox
        self.dpi = 90
        self.sliders = dict()
        self.controllers = dict()
        self.renderer = renderer
        self.plotter = self.renderer.plotter
        with self.disabled_interactivity():
            self.fig, self.dh = self.screenshot()
        self.configure_controllers()
        controllers = VBox(list(self.controllers.values()))
        layout = HBox([self.fig.canvas, controllers])
        display.display(layout)

    @contextmanager
    def disabled_interactivity(self):
        state = plt.isinteractive()
        plt.ioff()
        try:
            yield
        finally:
            if state:
                plt.ion()
            else:
                plt.ioff()

    def screenshot(self):
        width, height = self.renderer.figure.store['window_size']

        fig = plt.figure()
        fig.figsize = (width / self.dpi, height / self.dpi)
        fig.dpi = self.dpi
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.resizable = False
        fig.canvas.callbacks.callbacks.clear()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        dh = ax.imshow(self.plotter.screenshot())
        return fig, dh

    def update(self):
        self.plotter.render()
        self.dh.set_data(self.plotter.screenshot())
        self.fig.canvas.draw()

    def configure_controllers(self):
        from ipywidgets import (interactive, Label, VBox, FloatSlider,
                                IntSlider, Checkbox)
        # continuous update
        self.continuous_update_button = Checkbox(
            value=False,
            description='Continuous update',
            disabled=False,
            indent=False,
        )
        self.controllers["continuous_update"] = interactive(
            self.set_continuous_update,
            value=self.continuous_update_button
        )
        # subplot
        number_of_plots = len(self.plotter.renderers)
        if number_of_plots > 1:
            self.sliders["subplot"] = IntSlider(
                value=number_of_plots - 1,
                min=0,
                max=number_of_plots - 1,
                step=1,
                continuous_update=False
            )
            self.controllers["subplot"] = VBox([
                Label(value='Select the subplot'),
                interactive(
                    self.set_subplot,
                    index=self.sliders["subplot"],
                )
            ])
        # azimuth
        default_azimuth = self.plotter.renderer._azimuth
        self.sliders["azimuth"] = FloatSlider(
            value=default_azimuth,
            min=-180.,
            max=180.,
            step=10.,
            continuous_update=False
        )
        # elevation
        default_elevation = self.plotter.renderer._elevation
        self.sliders["elevation"] = FloatSlider(
            value=default_elevation,
            min=-180.,
            max=180.,
            step=10.,
            continuous_update=False
        )
        # distance
        eps = 1e-5
        default_distance = self.plotter.renderer._distance
        self.sliders["distance"] = FloatSlider(
            value=default_distance,
            min=eps,
            max=2. * default_distance - eps,
            step=default_distance / 10.,
            continuous_update=False
        )
        # camera
        self.controllers["camera"] = VBox([
            Label(value='Camera settings'),
            interactive(
                self.set_camera,
                azimuth=self.sliders["azimuth"],
                elevation=self.sliders["elevation"],
                distance=self.sliders["distance"],
            )
        ])

    def set_camera(self, azimuth, elevation, distance):
        focalpoint = self.plotter.camera.GetFocalPoint()
        self.renderer.set_camera(azimuth, elevation,
                                 distance, focalpoint)
        self.update()

    def set_subplot(self, index):
        row, col = self.plotter.index_to_loc(index)
        self.renderer.subplot(row, col)
        figure = self.renderer.figure
        default_azimuth = figure.plotter.renderer._azimuth
        default_elevation = figure.plotter.renderer._elevation
        default_distance = figure.plotter.renderer._distance
        self.sliders["azimuth"].value = default_azimuth
        self.sliders["elevation"].value = default_elevation
        self.sliders["distance"].value = default_distance

    def set_continuous_update(self, value):
        for slider in self.sliders.values():
            slider.continuous_update = value


_testing_context = nullcontext
