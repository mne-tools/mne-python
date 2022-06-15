"""Notebook implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import os
from contextlib import contextmanager, nullcontext

from IPython.display import display
from ipywidgets import (Button, Dropdown, FloatSlider, BoundedFloatText, HBox,
                        IntSlider, IntText, Text, VBox, IntProgress, Play,
                        Checkbox, RadioButtons, HTML, Accordion, jsdlink,
                        Layout, Select, GridBox)

from ._abstract import (_AbstractDock, _AbstractToolBar, _AbstractMenuBar,
                        _AbstractStatusBar, _AbstractLayout, _AbstractWidget,
                        _AbstractWindow, _AbstractMplCanvas, _AbstractPlayback,
                        _AbstractBrainMplCanvas, _AbstractMplInterface,
                        _AbstractWidgetList, _AbstractAction, _AbstractDialog,
                        _AbstractKeyPress)
from ._pyvista import _PyVistaRenderer, _close_all, _set_3d_view, _set_3d_title  # noqa: F401,E501, analysis:ignore


# modified from:
# https://gist.github.com/elkhadiy/284900b3ea8a13ed7b777ab93a691719
class _FilePicker:
    def __init__(self, rows=20, directory_only=False, ignore_dotfiles=True):
        self._callback = None
        self._directory_only = directory_only
        self._ignore_dotfiles = ignore_dotfiles
        self._empty_selection = True
        self._selected_dir = os.getcwd()
        self._item_layout = Layout(width='auto')
        self._nb_rows = rows
        self._file_selector = Select(
            options=self._get_selector_options(),
            rows=min(len(os.listdir(self._selected_dir)), self._nb_rows),
            layout=self._item_layout
        )
        self._open_button = Button(
            description='Open',
            layout=Layout(flex='auto 1 auto', width='auto')
        )
        self._select_button = Button(
            description='Select',
            layout=Layout(flex='auto 1 auto', width='auto')
        )
        self._cancel_button = Button(
            description='Cancel',
            layout=Layout(flex='auto 1 auto', width='auto')
        )
        self._parent_button = Button(
            icon='chevron-up',
            layout=Layout(flex='auto 1 auto', width='auto')
        )
        self._selection = Text(
            value=os.path.join(
                self._selected_dir, self._file_selector.value),
            disabled=True,
            layout=Layout(flex='1 1 auto', width='auto')
        )
        self._filename = Text(
            value='',
            layout=Layout(flex='1 1 auto', width='auto')
        )
        self._parent_button.on_click(self._parent_button_clicked)
        self._open_button.on_click(self._open_button_clicked)
        self._select_button.on_click(self._select_button_clicked)
        self._cancel_button.on_click(self._cancel_button_clicked)
        self._file_selector.observe(self._update_path)

        self._widget = VBox([
            HBox([
                self._parent_button, HTML(value='Look in:'), self._selection,
            ]),
            self._file_selector,
            HBox([
                HTML(value='File name'), self._filename, self._open_button,
                self._select_button, self._cancel_button,
            ]),
        ])

    def _get_selector_options(self):
        options = os.listdir(self._selected_dir)
        if self._ignore_dotfiles:
            tmp = list()
            for el in options:
                if el[0] != '.':
                    tmp.append(el)
            options = tmp
        if self._directory_only:
            tmp = list()
            for el in options:
                if os.path.isdir(os.path.join(self._selected_dir, el)):
                    tmp.append(el)
            options = tmp
        if not options:
            options = ['']
            self._empty_selection = True
        else:
            self._empty_selection = False
        return options

    def _update_selector_options(self):
        self._file_selector.options = self._get_selector_options()
        self._file_selector.rows = min(
            len(os.listdir(self._selected_dir)), self._nb_rows)
        self._selection.value = os.path.join(
            self._selected_dir, self._file_selector.value
        )
        self._filename.value = self._file_selector.value

    def show(self):
        self._update_selector_options()
        self._widget.layout.display = "block"

    def hide(self):
        self._widget.layout.display = "none"

    def set_directory_only(self, state):
        self._directory_only = state

    def set_ignore_dotfiles(self, state):
        self._ignore_dotfiles = state

    def connect(self, callback):
        self._callback = callback

    def _open_button_clicked(self, button):
        if self._empty_selection:
            return
        if os.path.isdir(self._selection.value):
            self._selected_dir = self._selection.value
            self._file_selector.options = self._get_selector_options()
            self._file_selector.rows = min(
                len(os.listdir(self._selected_dir)), self._nb_rows)

    def _select_button_clicked(self, button):
        if self._empty_selection:
            return
        result = os.path.join(self._selected_dir, self._filename.value)
        if self._callback is not None:
            self._callback(result)
            # the picker is shared so only one connection is allowed at a time
            self._callback = None  # reset the callback
        self.hide()

    def _cancel_button_clicked(self, button):
        self._callback = None  # reset the callback
        self.hide()

    def _parent_button_clicked(self, button):
        self._selected_dir, _ = os.path.split(self._selected_dir)
        self._update_selector_options()

    def _update_path(self, change):
        self._selection.value = os.path.join(
            self._selected_dir, self._file_selector.value
        )
        self._filename.value = self._file_selector.value


class _IpyKeyPress(_AbstractKeyPress):
    def _keypress_initialize(self, widget=None):
        pass

    def _keypress_add(self, shortcut, callback):
        pass

    def _keypress_trigger(self, shortcut):
        pass


class _IpyDialog(_AbstractDialog):
    def _dialog_create(self, title, text, info_text, callback, *,
                       icon='Warning', buttons=[], modal=True, window=None):
        pass


class _IpyLayout(_AbstractLayout):
    def _layout_initialize(self, max_width):
        self._layout_max_width = max_width

    def _layout_add_widget(self, layout, widget, stretch=0,
                           *, row=None, col=None):
        widget.layout.margin = "2px 0px 2px 0px"
        if not isinstance(widget, Play):
            widget.layout.min_width = "0px"
        if isinstance(layout, Accordion):
            box = layout.children[0]
        else:
            box = layout
        children = list(box.children)
        children.append(widget)
        box.children = tuple(children)
        # Fix columns
        if self._layout_max_width is not None and isinstance(widget, HBox):
            children = widget.children
            if len(children) > 0:
                width = int(self._layout_max_width / len(children))
                for child in children:
                    child.layout.width = f"{width}px"

    def _layout_create(self, orientation='vertical'):
        if orientation == 'vertical':
            layout = VBox()
        elif orientation == 'horizontal':
            layout = HBox()
        else:
            assert orientation == 'grid'
            layout = GridBox()
        return layout


class _IpyDock(_AbstractDock, _IpyLayout):
    def _dock_initialize(self, window=None, name="Controls",
                         area="left", max_width=None):
        if self._docks is None:
            self._docks = dict()
        current_dock = VBox()
        self._dock_width = 302
        self._dock = self._dock_layout = current_dock
        self._dock.layout.width = f"{self._dock_width}px"
        self._layout_initialize(self._dock_width)
        self._docks[area] = (self._dock, self._dock_layout)

    def _dock_finalize(self):
        pass

    def _dock_show(self):
        self._dock_layout.layout.visibility = "visible"

    def _dock_hide(self):
        self._dock_layout.layout.visibility = "hidden"

    def _dock_add_stretch(self, layout=None):
        pass

    def _dock_add_layout(self, vertical=True):
        return VBox() if vertical else HBox()

    def _dock_add_label(
        self, value, *, align=False, layout=None, selectable=False
    ):
        layout = self._dock_layout if layout is None else layout
        widget = HTML(value=value, disabled=True)
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_button(
        self, name, callback, *, style='pushbutton', icon=None, tooltip=None,
        layout=None
    ):
        layout = self._dock_layout if layout is None else layout
        kwargs = dict()
        if style == 'pushbutton':
            kwargs["description"] = name
        if tooltip is not None:
            kwargs["tooltip"] = tooltip
        widget = Button(**kwargs)
        widget.on_click(lambda x: callback())
        if icon is not None:
            widget.icon = icon
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_named_layout(self, name, *, layout=None, compact=True):
        layout = self._dock_layout if layout is None else layout
        if name is not None:
            hlayout = self._dock_add_layout(not compact)
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
            self._layout_add_widget(layout, hlayout)
            layout = hlayout
        return layout

    def _dock_add_slider(self, name, value, rng, callback, *,
                         compact=True, double=False, tooltip=None,
                         layout=None):
        layout = self._dock_named_layout(
            name=name, layout=layout, compact=compact)
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

    def _dock_add_check_box(self, name, value, callback, *, tooltip=None,
                            layout=None):
        layout = self._dock_layout if layout is None else layout
        widget = Checkbox(
            value=value,
            description=name,
            indent=False,
            disabled=False
        )
        hbox = HBox([widget])  # fix stretching to the right
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, hbox)
        return _IpyWidget(widget)

    def _dock_add_spin_box(self, name, value, rng, callback, *,
                           compact=True, double=True, step=None,
                           tooltip=None, layout=None):
        layout = self._dock_named_layout(
            name=name, layout=layout, compact=compact)
        klass = BoundedFloatText if double else IntText
        widget = klass(
            value=value,
            min=rng[0],
            max=rng[1],
        )
        if step is not None:
            widget.step = step
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_combo_box(self, name, value, rng, callback, *, compact=True,
                            tooltip=None, layout=None):
        layout = self._dock_named_layout(
            name=name, layout=layout, compact=compact)
        widget = Dropdown(
            value=value,
            options=rng,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_radio_buttons(self, value, rng, callback, *, vertical=True,
                                layout=None):
        # XXX: vertical=False is not supported yet
        layout = self._dock_layout if layout is None else layout
        widget = RadioButtons(
            options=rng,
            value=value,
            disabled=False,
        )
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidgetList(widget)

    def _dock_add_group_box(self, name, *, collapse=None, layout=None):
        layout = self._dock_layout if layout is None else layout
        if collapse is None:
            hlayout = VBox([HTML("<strong>" + name + "</strong>")])
        else:
            assert isinstance(collapse, bool)
            vbox = VBox()
            hlayout = Accordion([vbox])
            hlayout.set_title(0, name)
            if collapse:
                hlayout.selected_index = None
            else:
                hlayout.selected_index = 0
        self._layout_add_widget(layout, hlayout)
        return hlayout

    def _dock_add_text(self, name, value, placeholder, *, callback=None,
                       layout=None):
        layout = self._dock_layout if layout is None else layout
        widget = Text(value=value, placeholder=placeholder)
        if callback is not None:
            widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(layout, widget)
        return _IpyWidget(widget)

    def _dock_add_file_button(
        self, name, desc, func, *, filter=None, initial_directory=None,
        save=False, is_directory=False, icon=False, tooltip=None, layout=None
    ):
        layout = self._dock_layout if layout is None else layout

        def callback():
            self._file_picker.set_directory_only(is_directory)
            self._file_picker.connect(func)
            self._file_picker.show()

        if icon:
            kwargs = dict(style='toolbutton', icon='folder')
        else:
            kwargs = dict()
        widget = self._dock_add_button(
            name=desc,
            callback=callback,
            tooltip=tooltip,
            layout=layout,
            **kwargs
        )
        return widget


def _generate_callback(callback, to_float=False):
    def func(data):
        value = data["new"] if "new" in data else data["old"]
        callback(float(value) if to_float else value)
    return func


class _IpyToolBar(_AbstractToolBar, _IpyLayout):
    def _tool_bar_initialize(self, name="default", window=None):
        self.actions = dict()
        self._tool_bar = self._tool_bar_layout = HBox()
        self._layout_initialize(None)

    def _tool_bar_add_button(self, name, desc, func, *, icon_name=None,
                             shortcut=None):
        icon_name = name if icon_name is None else icon_name
        icon = self._icons[icon_name]
        if icon is None:
            return
        widget = Button(tooltip=desc, icon=icon)
        widget.on_click(lambda x: func())
        self._layout_add_widget(self._tool_bar_layout, widget)
        self.actions[name] = _IpyAction(widget)

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].set_icon(self._icons[icon_name])

    def _tool_bar_add_text(self, name, value, placeholder):
        widget = Text(value=value, placeholder=placeholder)
        self._layout_add_widget(self._tool_bar_layout, widget)
        self.actions[name] = _IpyAction(widget)

    def _tool_bar_add_spacer(self):
        pass

    def _tool_bar_add_file_button(self, name, desc, func, *, shortcut=None):
        def callback():
            fname = self.actions[f"{name}_field"].value
            func(None if len(fname) == 0 else fname)
        self._tool_bar_add_text(
            name=f"{name}_field",
            value=None,
            placeholder="Type a file name",
        )
        self._tool_bar_add_button(
            name=name,
            desc=desc,
            func=callback,
        )

    def _tool_bar_add_play_button(self, name, desc, func, *, shortcut=None):
        widget = Play(interval=500)
        self._layout_add_widget(self._tool_bar_layout, widget)
        self.actions[name] = _IpyAction(widget)
        return _IpyWidget(widget)


class _IpyMenuBar(_AbstractMenuBar):
    def _menu_initialize(self, window=None):
        self._menus = dict()
        self._menu_actions = dict()
        self._menu_desc2button = dict()  # only for notebook
        self._menu_bar = self._menu_bar_layout = HBox()
        self._layout_initialize(None)

    def _menu_add_submenu(self, name, desc):
        widget = Dropdown(value=desc, options=[desc])
        self._menus[name] = widget
        self._menu_actions[name] = dict()

        def callback(input_desc):
            if input_desc == desc:
                return
            button_name = self._menu_desc2button[input_desc]
            if button_name in self._menu_actions[name]:
                self._menu_actions[name][button_name].trigger()
            widget.value = desc
        widget.observe(_generate_callback(callback), names='value')
        self._layout_add_widget(self._menu_bar_layout, widget)

    def _menu_add_button(self, menu_name, name, desc, func):
        menu = self._menus[menu_name]
        options = list(menu.options)
        options.append(desc)
        menu.options = options
        self._menu_actions[menu_name][name] = _IpyAction(func)
        # associate the description with the name given by the user
        self._menu_desc2button[desc] = name


class _IpyStatusBar(_AbstractStatusBar, _IpyLayout):
    def _status_bar_initialize(self, window=None):
        self._status_bar = HBox()
        self._layout_initialize(None)

    def _status_bar_add_label(self, value, *, stretch=0):
        widget = Text(value=value, disabled=True)
        self._layout_add_widget(self._status_bar, widget)
        return _IpyWidget(widget)

    def _status_bar_add_progress_bar(self, stretch=0):
        widget = IntProgress()
        self._layout_add_widget(self._status_bar, widget)
        return _IpyWidget(widget)

    def _status_bar_update(self):
        pass


class _IpyPlayback(_AbstractPlayback):
    def _playback_initialize(self, func, timeout, value, rng,
                             time_widget, play_widget):
        play = play_widget._widget
        play.min = rng[0]
        play.max = rng[1]
        play.value = value
        slider = time_widget._widget
        jsdlink((play, 'value'), (slider, 'value'))
        jsdlink((slider, 'value'), (play, 'value'))


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
    def _window_initialize(
        self, *, window=None, central_layout=None, fullscreen=False
    ):
        super()._window_initialize()
        self._window_load_icons()

    def _window_load_icons(self):
        # from: https://fontawesome.com/icons
        self._icons["help"] = "question"
        self._icons["play"] = None
        self._icons["pause"] = None
        self._icons["reset"] = "history"
        self._icons["scale"] = "magic"
        self._icons["clear"] = "trash"
        self._icons["movie"] = "video-camera"
        self._icons["restore"] = "replay"
        self._icons["screenshot"] = "camera"
        self._icons["visibility_on"] = "eye"
        self._icons["visibility_off"] = "eye"
        self._icons["folder"] = "folder"

    def _window_close_connect(self, func, *, after=True):
        pass

    def _window_close_disconnect(self, after=True):
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

    def _window_new_cursor(self, name):
        pass

    @contextmanager
    def _window_ensure_minimum_sizes(self):
        yield

    def _window_set_theme(self, theme):
        pass

    def _window_create(self):
        pass
        # XXX: this could be a VBox if _Renderer.show is refactored


class _IpyWidgetList(_AbstractWidgetList):
    def __init__(self, src):
        self._src = src
        if isinstance(self._src, RadioButtons):
            self._widgets = _IpyWidget(self._src)
        else:
            self._widgets = list()
            for widget in self._src:
                if not isinstance(widget, _IpyWidget):
                    widget = _IpyWidget(widget)
                self._widgets.append(widget)

    def set_enabled(self, state):
        if isinstance(self._src, RadioButtons):
            self._widgets.set_enabled(state)
        else:
            for widget in self._widgets:
                widget.set_enabled(state)

    def get_value(self, idx):
        if isinstance(self._src, RadioButtons):
            # for consistency, we do not use get_value()
            return self._widgets._widget.options[idx]
        else:
            return self._widgets[idx].get_value()

    def set_value(self, idx, value):
        if isinstance(self._src, RadioButtons):
            self._widgets.set_value(value)
        else:
            self._widgets[idx].set_value(value)


class _IpyWidget(_AbstractWidget):
    def set_value(self, value):
        if isinstance(self._widget, Button):
            self._widget.click()
        else:
            self._widget.value = value

    def get_value(self):
        return self._widget.value

    def set_range(self, rng):
        self._widget.min = rng[0]
        self._widget.max = rng[1]

    def show(self):
        self._widget.layout.visibility = "visible"

    def hide(self):
        self._widget.layout.visibility = "hidden"

    def set_enabled(self, state):
        self._widget.disabled = not state

    def is_enabled(self):
        return not self._widget.disabled

    def update(self, repaint=True):
        pass

    def get_tooltip(self):
        assert hasattr(self._widget, 'tooltip')
        return self._widget.tooltip

    def set_tooltip(self, tooltip):
        assert hasattr(self._widget, 'tooltip')
        self._widget.tooltip = tooltip

    def set_style(self, style):
        for key, val in style.items():
            setattr(self._widget.layout, key, val)


class _IpyAction(_AbstractAction):
    def trigger(self):
        if callable(self._action):
            self._action()
        else:  # standard Button widget
            self._action.click()

    def set_icon(self, icon):
        self._action.icon = icon

    def set_shortcut(self, shortcut):
        pass


class _Renderer(_PyVistaRenderer, _IpyDock, _IpyToolBar, _IpyMenuBar,
                _IpyStatusBar, _IpyWindow, _IpyPlayback, _IpyDialog,
                _IpyKeyPress):
    _kind = 'notebook'

    def __init__(self, *args, **kwargs):
        self._docks = None
        self._menu_bar = None
        self._tool_bar = None
        self._status_bar = None
        self._file_picker = _FilePicker(rows=10)
        kwargs["notebook"] = True
        fullscreen = kwargs.pop('fullscreen', False)
        super().__init__(*args, **kwargs)
        self._window_initialize(fullscreen=fullscreen)

    def _update(self):
        if self.figure.display is not None:
            self.figure.display.update_canvas()

    def _display_default_tool_bar(self):
        self._tool_bar_initialize()
        self._tool_bar_add_file_button(
            name="screenshot",
            desc="Take a screenshot",
            func=self.screenshot,
        )
        display(self._tool_bar)

    def show(self):
        # menu bar
        if self._menu_bar is not None:
            display(self._menu_bar)
        # tool bar
        if self._tool_bar is not None:
            display(self._tool_bar)
        else:
            self._display_default_tool_bar()
        # viewer
        viewer = self.plotter.show(
            jupyter_backend="ipyvtklink", return_viewer=True)
        viewer.layout.width = None  # unlock the fixed layout
        rendering_row = list()
        if self._docks is not None and "left" in self._docks:
            rendering_row.append(self._docks["left"][0])
        rendering_row.append(viewer)
        if self._docks is not None and "right" in self._docks:
            rendering_row.append(self._docks["right"][0])
        display(HBox(rendering_row))
        self.figure.display = viewer
        # status bar
        if self._status_bar is not None:
            display(self._status_bar)
        # file picker
        self._file_picker.hide()
        display(self._file_picker._widget)
        return self.scene()


_testing_context = nullcontext
