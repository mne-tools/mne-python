"""Notebook implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
from IPython.display import display
from ipywidgets import (Button, Dropdown, FloatSlider, FloatText, HBox,
                        IntSlider, IntText, Text, VBox)

from ..utils import _save_ndarray_img
from ...fixes import nullcontext
from ._abstract import (_AbstractDock, _AbstractToolBar, _AbstractMenuBar,
                        _AbstractStatusBar, _AbstractLayout, _AbstractWidget,
                        _AbstractWindow, _AbstractMplCanvas, _AbstractPlayback,
                        _AbstractBrainMplCanvas, _AbstractMplInterface)
from ._pyvista import _PyVistaRenderer, _close_all, _set_3d_view, _set_3d_title  # noqa: F401,E501, analysis:ignore


class _IpyLayout(_AbstractLayout):
    def _layout_initialize(self, max_width):
        self._layout_max_width = max_width

    def _layout_add_widget(self, layout, widget):
        widget.layout.margin = "2px 0px 2px 0px"
        widget.layout.min_width = "0px"
        children = list(layout.children)
        children.append(widget)
        layout.children = tuple(children)
        # Fix columns
        if self._layout_max_width is not None and isinstance(widget, HBox):
            children = widget.children
            width = int(self._layout_max_width / len(children))
            for child in children:
                child.layout.width = f"{width}px"


class _IpyDock(_AbstractDock, _IpyLayout):
    def _dock_initialize(self, window=None):
        self.dock_width = 300
        self.dock = self.dock_layout = VBox()
        self.dock.layout.width = f"{self.dock_width}px"
        self._layout_initialize(self.dock_width)

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
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_button(self, name, callback, layout=None):
        widget = Button(description=name)
        widget.on_click(lambda x: callback())
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_named_layout(self, name, layout, compact):
        layout = self.dock_layout if layout is None else layout
        if name is not None:
            hlayout = self._dock_add_layout(not compact)
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
            self._layout_add_widget(layout, hlayout)
            layout = hlayout
        return layout

    def _dock_add_slider(self, name, value, rng, callback,
                         compact=True, double=False, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        klass = FloatSlider if double else IntSlider
        widget = klass(
            value=value,
            min=rng[0],
            max=rng[1],
            readout=False,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_spin_box(self, name, value, rng, callback,
                           compact=True, double=True, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        klass = FloatText if double else IntText
        widget = klass(
            value=value,
            min=rng[0],
            max=rng[1],
            readout=False,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        widget = Dropdown(
            value=value,
            options=rng,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_group_box(self, name, layout=None):
        layout = self.dock_layout if layout is None else layout
        hlayout = VBox()
        self._layout_add_widget(layout, hlayout)
        return hlayout


def _generate_callback(callback, to_float=False):
    def func(data):
        value = data["new"] if "new" in data else data["old"]
        callback(float(value) if to_float else value)
    return func


class _IpyToolBar(_AbstractToolBar, _IpyLayout):
    def _tool_bar_load_icons(self):
        self.icons = dict()
        self.icons["help"] = "question"
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

    def _tool_bar_initialize(self, name="default", window=None):
        self.actions = dict()
        self.tool_bar = HBox()
        self._layout_initialize(None)

    def _tool_bar_add_button(self, name, desc, func, icon_name=None,
                             shortcut=None):
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        if icon is None:
            return
        widget = Button(tooltip=desc, icon=icon)
        widget.on_click(lambda x: func())
        self._layout_add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].icon = self.icons[icon_name]

    def _tool_bar_add_text(self, name, value, placeholder):
        widget = Text(value=value, placeholder=placeholder)
        self._layout_add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_add_spacer(self):
        pass

    def _tool_bar_add_screenshot_button(self, name, desc, func):
        def _screenshot():
            fname = self.actions[f"{name}_field"].value
            fname = self._get_screenshot_filename() \
                if len(fname) == 0 else fname
            img = func()
            _save_ndarray_img(fname, img)

        self._tool_bar_add_button(
            name=name,
            desc=desc,
            func=_screenshot,
        )
        self._tool_bar_add_text(
            name=f"{name}_field",
            value=None,
            placeholder="Type a file name",
        )

    def _tool_bar_set_theme(self, theme):
        pass


class _IpyMenuBar(_AbstractMenuBar):
    def _menu_initialize(self, window=None):
        pass

    def _menu_add_submenu(self, name, desc):
        pass

    def _menu_add_button(self, menu_name, name, desc, func):
        pass


class _IpyStatusBar(_AbstractStatusBar):
    def _status_bar_initialize(self, window=None):
        pass

    def _status_bar_add_label(self, value, stretch=0):
        pass

    def _status_bar_add_progress_bar(self, stretch=0):
        pass


class _IpyPlayback(_AbstractPlayback):
    def _playback_initialize(self, func, timeout):
        pass


class _IpyMplInterface(_AbstractMplInterface):
    def _mpl_initialize(self):
        from matplotlib.backends.backend_nbagg import (FigureCanvasNbAgg,
                                                       FigureManager)
        self.canvas = FigureCanvasNbAgg(self.fig)
        self.manager = FigureManager(self.canvas, 0)


class _IpyMplCanvas(_AbstractMplCanvas, _IpyMplInterface):
    def __init__(self, width, height, dpi):
        super().__init__(width, height, dpi)
        self._mpl_initialize()


class _IpyBrainMplCanvas(_AbstractBrainMplCanvas, _IpyMplInterface):
    def __init__(self, brain, width, height, dpi):
        super().__init__(brain, width, height, dpi)
        self._mpl_initialize()
        self._connect()


class _IpyWindow(_AbstractWindow):
    def _window_close_connect(self, func):
        pass

    def _window_get_dpi(self):
        return 96

    def _window_get_size(self):
        return self.figure.plotter.window_size

    def _window_get_simple_canvas(self, width, height, dpi):
        return _IpyMplCanvas(width, height, dpi)

    def _window_get_mplcanvas(self, brain, interactor_fraction, show_traces,
                              separate_canvas):
        w, h = self._window_get_mplcanvas_size(interactor_fraction)
        self._interactor_fraction = interactor_fraction
        self._show_traces = show_traces
        self._separate_canvas = separate_canvas
        self._mplcanvas = _IpyBrainMplCanvas(
            brain, w, h, self._window_get_dpi())
        return self._mplcanvas

    def _window_adjust_mplcanvas_layout(self):
        pass

    def _window_get_cursor(self):
        pass

    def _window_set_cursor(self, cursor):
        pass

    @contextmanager
    def _window_ensure_minimum_sizes(self):
        yield

    def _window_set_theme(self, theme):
        pass


class _IpyWidget(_AbstractWidget):
    def set_value(self, value):
        self._widget.value = value

    def get_value(self):
        return self._widget.value


class _Renderer(_PyVistaRenderer, _IpyDock, _IpyToolBar, _IpyMenuBar,
                _IpyStatusBar, _IpyWindow, _IpyPlayback):
    def __init__(self, *args, **kwargs):
        self.dock = None
        self.tool_bar = None
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def _update(self):
        if self.figure.display is not None:
            self.figure.display.update_canvas()

    def _create_default_tool_bar(self):
        self._tool_bar_load_icons()
        self._tool_bar_initialize()
        self._tool_bar_add_screenshot_button(
            name="screenshot",
            desc="Take a screenshot",
            func=self.screenshot,
        )

    def show(self):
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
