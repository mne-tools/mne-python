# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ..backends._notebook \
    import _NotebookInteractor as _PyVistaNotebookInteractor


class _NotebookInteractor(_PyVistaNotebookInteractor):
    def __init__(self, brain):
        self.brain = brain
        super().__init__(self.brain._renderer)

    def configure_controllers(self):
        from ipywidgets import (IntSlider, interactive, Play, VBox,
                                HBox, Label, jslink)
        super().configure_controllers()
        # orientation
        self.controllers["orientation"] = interactive(
            self.set_orientation,
            orientation=self.brain.orientation,
        )
        # smoothing
        self.sliders["smoothing"] = IntSlider(
            value=self.brain._data['smoothing_steps'],
            min=self.brain.default_smoothing_range[0],
            max=self.brain.default_smoothing_range[1],
            continuous_update=False
        )
        self.controllers["smoothing"] = VBox([
            Label(value='Smoothing steps'),
            interactive(
                self.brain.set_data_smoothing,
                n_steps=self.sliders["smoothing"]
            )
        ])
        # time slider
        max_time = len(self.brain._data['time']) - 1
        if max_time >= 1:
            time_player = Play(
                value=self.brain._data['time_idx'],
                min=0,
                max=max_time,
                continuous_update=False
            )
            time_slider = IntSlider(
                min=0,
                max=max_time,
            )
            jslink((time_player, 'value'), (time_slider, 'value'))
            time_slider.observe(self.set_time_point, 'value')
            self.controllers["time"] = VBox([
                HBox([
                    Label(value='Select time point'),
                    time_player,
                ]),
                time_slider,
            ])
            self.sliders["time"] = time_slider

    def set_orientation(self, orientation):
        row, col = self.plotter.index_to_loc(
            self.plotter._active_renderer_index)
        self.brain.show_view(orientation, row=row, col=col)

    def set_time_point(self, data):
        self.brain.set_time_point(data['new'])
