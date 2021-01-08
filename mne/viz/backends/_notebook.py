# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ...fixes import nullcontext
from ._pyvista import _Renderer as _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer):
    def __init__(self, *args, **kwargs):
        self.default_screenshot_name = "screenshot.png"
        self.tool_bar_state = True
        self.tool_bar = None
        self.actions = dict()
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def _screenshot(self):
        fname = self.actions.get("screenshot_field").value
        fname = self.default_screenshot_name if len(fname) == 0 else fname
        self.screenshot(filename=fname)

    def _set_tool_bar(self, state):
        self.tool_bar_state = state

    def _add_button(self, desc, func, icon_name):
        from ipywidgets import Button
        button = Button(tooltip=desc, icon=icon_name)
        button.on_click(lambda x: func())
        return button

    def _add_text_field(self, value, placeholder):
        from ipywidgets import Text
        return Text(value=value, placeholder=placeholder)

    def _show_tool_bar(self, actions):
        from IPython import display
        from ipywidgets import HBox
        tool_bar = HBox(tuple(actions.values()))
        display.display(tool_bar)
        return tool_bar

    def _configure_tool_bar(self):
        self.actions["screenshot"] = self._add_button(
            desc="Take a screenshot",
            func=self._screenshot,
            icon_name="camera",
        )
        self.actions["screenshot_field"] = self._add_text_field(
            value="screenshot.png",
            placeholder="Type file name",
        )
        self.tool_bar = self._show_tool_bar(self.actions)

    def show(self):
        from IPython.display import display
        if self.tool_bar_state:
            self._configure_tool_bar()
        self.figure.display = self.plotter.show(use_ipyvtk=True,
                                                return_viewer=True)
        self.figure.display.layout.width = None  # unlock the fixed layout
        display(self.figure.display)
        return self.scene()


_testing_context = nullcontext
