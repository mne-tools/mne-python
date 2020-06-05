# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ..backends._pyvista import _JupyterInteractor


class _BrainJupyterInteractor(_JupyterInteractor):
    def __init__(self, brain):
        self.brain = brain
        super().__init__(brain._renderer)

    def configure_controllers(self):
        from ipywidgets import FloatSlider, VBox, interactive
        super().configure_controllers()
        # time slider
        max_time = len(self.brain._data['time']) - 1
        if max_time >= 1:
            self.sliders["time"] = FloatSlider(
                value=self.brain._data['time_idx'],
                min=0,
                max=max_time,
                continuous_update=False
            )
            self.controllers["time"] = VBox([
                interactive(
                    self.brain.set_time_point,
                    time_idx=self.sliders["time"],
                )
            ])
