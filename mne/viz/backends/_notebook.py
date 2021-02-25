# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ...fixes import nullcontext
from ._pyvista import _Renderer as _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer):
    def __init__(self, *args, **kwargs):
        self.tool_bar = None
        self.dock_width = 300
        self.dock = None
        self.actions = None
        self.widgets = None
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

    def _add_tool_bar_button(self, desc, func, icon_name):
        from ipywidgets import Button
        button = Button(tooltip=desc, icon=icon_name)
        button.on_click(lambda x: func())
        return button

    def _add_tool_bar_text(self, value, placeholder):
        from ipywidgets import Text
        return Text(value=value, placeholder=placeholder)

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
            _generate_callback(callback, to_float=validator))
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
        widget.observe(_generate_callback(callback))
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
        widget.observe(_generate_callback(callback))
        self._add_widget(hlayout, widget)
        self._add_widget(layout, hlayout)
        return widget

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
        from ipywidgets import Combobox
        layout = self.dock_layout if layout is None else layout
        hlayout = self._dock_add_layout(not compact)
        if name is not None:
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
        widget = Combobox(
            value=value,
            options=rng,
        )
        widget.observe(_generate_callback(callback))
        self._add_widget(hlayout, widget)
        self._add_widget(layout, hlayout)
        return widget

    def _dock_add_group_box(self, name, layout=None):
        from ipywidgets import VBox
        layout = self.dock_layout if layout is None else layout
        hlayout = VBox()
        self._add_widget(layout, hlayout)
        return hlayout

    def _initialize_tool_bar(self, actions=None):
        if actions is None:
            actions = dict()
            actions["screenshot"] = self._add_tool_bar_button(
                desc="Take a screenshot",
                func=self._screenshot,
                icon_name="camera",
            )
            actions["screenshot_field"] = self._add_tool_bar_text(
                value=None,
                placeholder="Type a file name",
            )
        self.actions = actions

    def _finalize_tool_bar(self):
        if self.actions is None:
            return None
        from IPython import display
        from ipywidgets import HBox
        tool_bar = HBox(tuple(self.actions.values()))
        display.display(tool_bar)
        return tool_bar

    def show(self):
        from ipywidgets import HBox
        from IPython.display import display
        # tool bar
        if self.actions is None:
            self._initialize_tool_bar()
        self.tool_bar = self._finalize_tool_bar()
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
        if isinstance(data["new"], dict):
            if "value" in data["new"]:
                value = data["new"]["value"]
            else:
                value = data["old"]["value"]
            callback(float(value) if to_float else value)
    return func


_testing_context = nullcontext
