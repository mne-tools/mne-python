# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ...fixes import nullcontext
from ._ipy_dock import _IpyDock
from ._ipy_tool_bar import _IpyToolBar
from ._pyvista import _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer, _IpyDock, _IpyToolBar):
    def __init__(self, *args, **kwargs):
        self.dock = None
        self.tool_bar = None
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def _screenshot(self):
        fname = self.actions["screenshot_field"].value
        fname = self._get_screenshot_filename() if len(fname) == 0 else fname
        self.screenshot(filename=fname)

    def _create_default_tool_bar(self):
        self._tool_bar_load_icons()
        self._tool_bar_initialize()
        self._tool_bar_add_button(
            name="screenshot",
            desc="Take a screenshot",
            func=self._screenshot,
        )
        self._tool_bar_add_text(
            name="screenshot_field",
            value=None,
            placeholder="Type a file name",
        )

    def show(self):
        from ipywidgets import HBox
        from IPython.display import display
        # default tool bar
        if self.tool_bar is None:
            self._create_default_tool_bar()
        display(self.tool_bar)
        # viewer
        viewer = self.plotter.show(
            use_ipyvtk=True, return_viewer=True)
        viewer.layout.width = None  # unlock the fixed layout
        # main widget
        if self.dock is None:
            main_widget = viewer
        else:
            main_widget = HBox([self.dock, viewer])
        display(main_widget)
        self.figure.display = viewer
        return self.scene()


_testing_context = nullcontext
