# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ...fixes import nullcontext
from .abstract_dock import _AbstractDock
from .abstract_tool_bar import _AbstractToolBar
from ._pyvista import _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer, _AbstractDock, _AbstractToolBar):
    def __init__(self, *args, **kwargs):
        self.tool_bar = None
        self.dock_width = 300
        self.dock = None
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def _add_widget(self, layout, widget):
        from ipywidgets import HBox
        widget.layout.margin = "2px 0px 2px 0px"
        widget.layout.min_width = "0px"
        children = list(layout.children)
        children.append(widget)
        layout.children = tuple(children)
        # Fix columns
        if isinstance(widget, HBox):
            children = widget.children
            width = int(self.dock_width / len(children))
            for child in children:
                child.layout.width = f"{width}px"

    def _screenshot(self):
        fname = self.actions.get("screenshot_field").value
        fname = self._get_screenshot_filename() if len(fname) == 0 else fname
        self.screenshot(filename=fname)

    def _dock_initialize(self):
        from ipywidgets import VBox
        self.dock = self.dock_layout = VBox()
        self.dock.layout.width = f"{self.dock_width}px"

    def _dock_finalize(self):
        pass

    def _dock_show(self):
        self.dock_layout.layout.visibility = "visible"

    def _dock_hide(self):
        self.dock_layout.layout.visibility = "hidden"

    def _dock_add_stretch(self, layout):
        pass

    def _dock_add_layout(self, vertical=True):
        from ipywidgets import VBox, HBox
        return VBox() if vertical else HBox()

    def _dock_add_label(self, value, align=False, layout=None):
        from ipywidgets import Text
        layout = self.dock_layout if layout is None else layout
        widget = Text(value=value, disabled=True)
        self._add_widget(layout, widget)
        return widget

    def _dock_add_button(self, name, callback, layout=None):
        from ipywidgets import Button
        widget = Button(description=name)
        widget.on_click(lambda x: callback())
        self._add_widget(layout, widget)
        return widget

    def _dock_add_text(self, value, callback, validator=None,
                       layout=None):
        from ipywidgets import Text
        layout = self.dock_layout if layout is None else layout
        widget = Text(value=value)
        widget.observe(
            _generate_callback(callback, to_float=validator), names='value')
        self._add_widget(layout, widget)
        return widget

    def _dock_add_slider(self, name, value, rng, callback,
                         compact=True, double=False, layout=None):
        from ipywidgets import IntSlider, FloatSlider
        layout = self.dock_layout if layout is None else layout
        hlayout = self._dock_add_layout(not compact)
        if name is not None:
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
        klass = FloatSlider if double else IntSlider
        widget = klass(
            value=value,
            min=rng[0],
            max=rng[1],
            readout=False,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._add_widget(hlayout, widget)
        self._add_widget(layout, hlayout)
        return widget

    def _dock_add_spin_box(self, name, value, rng, callback,
                           compact=True, double=True, layout=None):
        from ipywidgets import IntText, FloatText
        layout = self.dock_layout if layout is None else layout
        hlayout = self._dock_add_layout(not compact)
        if name is not None:
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
        klass = FloatText if double else IntText
        widget = klass(
            value=value,
            min=rng[0],
            max=rng[1],
            readout=False,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._add_widget(hlayout, widget)
        self._add_widget(layout, hlayout)
        return widget

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
        from ipywidgets import Dropdown
        layout = self.dock_layout if layout is None else layout
        hlayout = self._dock_add_layout(not compact)
        if name is not None:
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
        widget = Dropdown(
            value=value,
            options=rng,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._add_widget(hlayout, widget)
        self._add_widget(layout, hlayout)
        return widget

    def _dock_add_group_box(self, name, layout=None):
        from ipywidgets import VBox
        layout = self.dock_layout if layout is None else layout
        hlayout = VBox()
        self._add_widget(layout, hlayout)
        return hlayout

    def _tool_bar_load_icons(self):
        self.icons = dict()
        self.icons["help"] = None
        self.icons["play"] = None
        self.icons["pause"] = None
        self.icons["reset"] = "history"
        self.icons["scale"] = "magic"
        self.icons["clear"] = "trash"
        self.icons["movie"] = None
        self.icons["restore"] = "replay"
        self.icons["screenshot"] = "camera"
        self.icons["visibility_on"] = "eye"
        self.icons["visibility_off"] = "eye"

    def _tool_bar_initialize(self, name="default"):
        from ipywidgets import HBox
        self.actions = dict()
        self.tool_bar = HBox()

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

    def _tool_bar_finalize(self):
        pass

    def _tool_bar_add_button(self, name, desc, func, icon_name=None):
        from ipywidgets import Button
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        if icon is None:
            return
        widget = Button(tooltip=desc, icon=icon)
        widget.on_click(lambda x: func())
        self._add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].icon = self.icons[icon_name]

    def _tool_bar_add_text(self, name, value, placeholder):
        from ipywidgets import Text
        widget = Text(value=value, placeholder=placeholder)
        self._add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_add_spacer(self):
        pass

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


def _generate_callback(callback, to_float=False):
    def func(data):
        value = data["new"] if "new" in data else data["old"]
        callback(float(value) if to_float else value)
    return func


_testing_context = nullcontext
