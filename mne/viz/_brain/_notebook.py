# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ..backends._notebook \
    import _NotebookInteractor as _PyVistaNotebookInteractor


class _NotebookInteractor(_PyVistaNotebookInteractor):
    def __init__(self, time_viewer):
        self.time_viewer = time_viewer
        self.brain = self.time_viewer.brain
        super().__init__(self.brain._renderer)

    def configure_controllers(self):
        from ipywidgets import IntSlider, FloatSlider, interactive
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
            self.controllers["time"] = interactive(
                self.brain.set_time_point,
                time_idx=self.sliders["time"],
            )
        # orientation
        self.controllers["orientation"] = interactive(
            self.set_orientation,
            orientation=self.time_viewer.orientation,
        )
        # smoothing
        self.sliders["smoothing"] = IntSlider(
            value=self.brain._data['smoothing_steps'],
            min=self.time_viewer.default_smoothing_range[0],
            max=self.time_viewer.default_smoothing_range[1],
            continuous_update=False
        )
        self.controllers["smoothing"] = interactive(
            self.brain.set_data_smoothing,
            n_steps=self.sliders["smoothing"]
        )

    def set_orientation(self, orientation):
        row, col = self.plotter.index_to_loc(
            self.plotter._active_renderer_index)
        self.brain.show_view(orientation, row=row, col=col)
