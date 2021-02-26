"""Dock implemented with ipywidgets."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from .abstract_dock import _AbstractDock
from ._utils import _ipy_add_widget
from ipywidgets import (Button, Dropdown, FloatSlider, FloatText, HBox,
                        IntSlider, IntText, Text, VBox)


class _IpyDock(_AbstractDock):
    def _dock_initialize(self):
        self.dock_width = 300
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
        return VBox() if vertical else HBox()

    def _dock_add_label(self, value, align=False, layout=None):
        layout = self.dock_layout if layout is None else layout
        widget = Text(value=value, disabled=True)
        _ipy_add_widget(layout, widget, self.dock_width)
        return widget

    def _dock_add_button(self, name, callback, layout=None):
        widget = Button(description=name)
        widget.on_click(lambda x: callback())
        _ipy_add_widget(layout, widget, self.dock_width)
        return widget

    def _dock_add_text(self, value, callback, validator=None,
                       layout=None):
        layout = self.dock_layout if layout is None else layout
        widget = Text(value=value)
        widget.observe(
            _generate_callback(callback, to_float=validator), names='value')
        _ipy_add_widget(layout, widget, self.dock_width)
        return widget

    def _dock_add_slider(self, name, value, rng, callback,
                         compact=True, double=False, layout=None):
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
        _ipy_add_widget(hlayout, widget, self.dock_width)
        _ipy_add_widget(layout, hlayout, self.dock_width)
        return widget

    def _dock_add_spin_box(self, name, value, rng, callback,
                           compact=True, double=True, layout=None):
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
        _ipy_add_widget(hlayout, widget, self.dock_width)
        _ipy_add_widget(layout, hlayout, self.dock_width)
        return widget

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
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
        _ipy_add_widget(hlayout, widget, self.dock_width)
        _ipy_add_widget(layout, hlayout, self.dock_width)
        return widget

    def _dock_add_group_box(self, name, layout=None):
        layout = self.dock_layout if layout is None else layout
        hlayout = VBox()
        _ipy_add_widget(layout, hlayout, self.dock_width)
        return hlayout


def _generate_callback(callback, to_float=False):
    def func(data):
        value = data["new"] if "new" in data else data["old"]
        callback(float(value) if to_float else value)
    return func
