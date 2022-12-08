"""Notebook implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

import os
import os.path as op
from contextlib import contextmanager, nullcontext

from IPython.display import display, clear_output
from ipywidgets import (Widget, HBox, VBox, Button, Dropdown, IntSlider,
                        IntText, Text, IntProgress, Play, Label, HTML,
                        Checkbox, RadioButtons, Accordion, link,
                        Layout, Select, GridBox,
                        # non-object-based-abstraction-only widgets, deprecate
                        FloatSlider, BoundedFloatText, jsdlink)
from ipyevents import Event

from ._abstract import (_AbstractAppWindow, _AbstractHBoxLayout,
                        _AbstractVBoxLayout, _AbstractGridLayout,
                        _AbstractWidget, _AbstractCanvas,
                        _AbstractPopup, _AbstractLabel,
                        _AbstractButton, _AbstractSlider,
                        _AbstractCheckBox, _AbstractSpinBox, _AbstractComboBox,
                        _AbstractRadioButtons, _AbstractGroupBox,
                        _AbstractText, _AbstractFileButton,
                        _AbstractPlayMenu, _AbstractProgressBar)
from ._abstract import (_AbstractDock, _AbstractToolBar, _AbstractMenuBar,
                        _AbstractStatusBar, _AbstractLayout, _AbstractWdgt,
                        _AbstractWindow, _AbstractMplCanvas, _AbstractPlayback,
                        _AbstractBrainMplCanvas, _AbstractMplInterface,
                        _AbstractWidgetList, _AbstractAction, _AbstractDialog,
                        _AbstractKeyPress)
from ._pyvista import _PyVistaRenderer, Plotter
from ._pyvista import (_close_3d_figure, _check_3d_figure, _close_all,  # noqa: F401,E501 analysis:ignore
                       _set_3d_view, _set_3d_title, _take_3d_screenshot)  # noqa: F401,E501 analysis:ignore
from ._utils import _notebook_vtk_works


# dict values are icon names from: https://fontawesome.com/icons
_ICON_LUT = dict(help='question', play='play', pause='pause',
                 reset='history', scale='magic',
                 clear='trash', movie='video-camera',
                 restore='replay', screenshot='camera',
                 visibility_on='eye', visibility_off='eye',
                 folder='folder', question='question',
                 information='info', warning='triangle-exclamation',
                 critical='exclamation')

_BASE_MIN_SIZE = '20px'
_BASE_KWARGS = dict(layout=Layout(
    min_width=_BASE_MIN_SIZE, min_height=_BASE_MIN_SIZE))

# %%
# Widgets
# -------
# The metaclasses need to share a base class in order for the inheritance
# not to conflict, http://www.phyast.pitt.edu/~micheles/python/metatype.html
# https://stackoverflow.com/questions/28720217/multiple-inheritance-metaclass-conflict


class _BaseWidget(type(_AbstractWidget), type(Widget)):
    pass


class _Widget(Widget, _AbstractWidget, metaclass=_BaseWidget):

    tooltip = None

    def __init__(self):
        _AbstractWidget.__init__()
        # Widget cannot init because the layouts (HBox, VBox and GridBox) don't
        # inherit from Widget like they do analogously for Qt, this isn't an
        # issue since each subclass __init__s it's own (e.g. Label)
        # Widget.__init__(self)

    def _set_range(self, rng):
        self.min = rng[0]
        self.max = rng[1]

    def _show(self):
        self.layout.visibility = "visible"

    def _hide(self):
        self.layout.visibility = "hidden"

    def _set_enabled(self, state):
        self.disabled = not state

    def _is_enabled(self):
        return not self.disabled

    def _update(self, repaint=True):
        pass

    def _get_tooltip(self):
        return self.tooltip

    def _set_tooltip(self, tooltip):
        self.tooltip = tooltip

    def _set_style(self, style):
        for key, val in style.items():
            setattr(self.layout, key, val)

    def _add_keypress(self, callback):
        self._event_watcher = Event(source=self, watched_events=['keydown'])
        self._event_watcher.on_dom_event(
            lambda event: callback(event['key'].lower().replace('arrow', '')))
        self._callback = callback

    def _trigger_keypress(self, key):
        # note: this doesn't actually simulate a keypress, it just calls the
        # callback function directly because this is not yet possible
        self._callback(key)

    def _set_focus(self):
        if hasattr(self, 'focus'):  # added in ipywidgets 8.0
            self.focus()

    def _set_layout(self, layout):
        self.children = (layout,)

    def _set_theme(self, theme):
        pass

    def _set_size(self, width=None, height=None):
        if width:
            self.layout.width = width
        if height:
            self.layout.height = height


class _Label(_Widget, _AbstractLabel, Label, metaclass=_BaseWidget):

    def __init__(self, value, center=False, selectable=False):
        _Widget.__init__(self)
        _AbstractLabel.__init__(value, center=center, selectable=selectable)
        kwargs = _BASE_KWARGS.copy()
        if center:
            kwargs['layout'].justify_content = 'center'
        Label.__init__(self, value=value, disabled=True, **kwargs)


class _Text(_AbstractText, _Widget, Text, metaclass=_BaseWidget):

    def __init__(self, value=None, placeholder=None, callback=None):
        _AbstractText.__init__(value=value, placeholder=placeholder,
                               callback=callback)
        _Widget.__init__(self)
        Text.__init__(self, value=value, placeholder=placeholder,
                      **_BASE_KWARGS)
        if callback is not None:
            self.observe(lambda x: callback(x['new']), names='value')

    def _set_value(self, value):
        self.value = value


class _Button(_Widget, _AbstractButton, Button, metaclass=_BaseWidget):

    def __init__(self, value, callback, icon=None):
        _Widget.__init__(self)
        _AbstractButton.__init__(value=value, callback=callback)
        Button.__init__(self, description=value, **_BASE_KWARGS)
        self.on_click(lambda x: callback())
        if icon:
            self.icon = _ICON_LUT[icon]

    def _click(self):
        self.click()

    def _set_icon(self, icon):
        self.icon = _ICON_LUT[icon]


class _Slider(_Widget, _AbstractSlider, IntSlider, metaclass=_BaseWidget):

    def __init__(self, value, rng, callback, horizontal=True):
        _Widget.__init__(self)
        _AbstractSlider.__init__(value=value, rng=rng, callback=callback,
                                 horizontal=horizontal)
        IntSlider.__init__(
            self, value=int(value), min=int(rng[0]), max=int(rng[1]),
            readout=False,
            orientation='horizontal' if horizontal else 'vertical',
            **_BASE_KWARGS)
        self.observe(lambda x: callback(x['new']), names='value')

    def _set_value(self, value):
        self.value = value

    def _get_value(self):
        return self.value

    def set_range(self, rng):
        self.min = int(rng[0])
        self.max = int(rng[1])


class _ProgressBar(_AbstractProgressBar, _Widget, IntProgress,
                   metaclass=_BaseWidget):

    def __init__(self, count):
        _AbstractProgressBar.__init__(count=count)
        _Widget.__init__(self)
        IntProgress.__init__(self, max=count, **_BASE_KWARGS)

    def _increment(self):
        if self.value + 1 > self.max:
            return
        self.value += 1
        return self.value


class _CheckBox(_Widget, _AbstractCheckBox, Checkbox, metaclass=_BaseWidget):

    def __init__(self, value, callback):
        _Widget.__init__(self)
        _AbstractCheckBox.__init__(value=value, callback=callback)
        Checkbox.__init__(self, value=value, **_BASE_KWARGS)
        self.observe(lambda x: callback(x['new']), names='value')

    def _set_checked(self, checked):
        self.value = checked

    def _get_checked(self):
        return self.value


class _SpinBox(_Widget, _AbstractSpinBox, IntText, metaclass=_BaseWidget):

    def __init__(self, value, rng, callback, step=None):
        _Widget.__init__(self)
        _AbstractSpinBox.__init__(value=value, rng=rng, callback=callback,
                                  step=step)
        IntText.__init__(self, value=value, min=rng[0], max=rng[1],
                         **_BASE_KWARGS)
        if step is not None:
            self.step = step
        self.observe(lambda x: callback(x['new']), names='value')

    def _set_value(self, value):
        self.value = value

    def _get_value(self):
        return self.value


class _ComboBox(_AbstractComboBox, _Widget, Dropdown, metaclass=_BaseWidget):

    def __init__(self, value, items, callback):
        _AbstractComboBox.__init__(value=value, items=items, callback=callback)
        _Widget.__init__(self)
        Dropdown.__init__(self, value=value, options=items, **_BASE_KWARGS)
        self.observe(lambda x: callback(x['new']), names='value')

    def _set_value(self, value):
        self.value = value

    def _get_value(self):
        return self.value


class _RadioButtons(_AbstractRadioButtons, _Widget, RadioButtons,
                    metaclass=_BaseWidget):

    def __init__(self, value, items, callback):
        _AbstractRadioButtons.__init__(
            value=value, items=items, callback=callback)
        _Widget.__init__(self)
        RadioButtons.__init__(self, value=value, options=items,
                              disabled=False, **_BASE_KWARGS)
        self.observe(lambda x: callback(x['new']), names='value')

    def _set_value(self, value):
        self.value = value

    def _get_value(self):
        return self.value


class _GroupBox(_AbstractGroupBox, _Widget, Accordion, metaclass=_BaseWidget):

    def __init__(self, name, items):
        _AbstractGroupBox.__init__(name=name, items=items)
        _Widget.__init__(self)
        kwargs = _BASE_KWARGS.copy()
        kwargs['layout'].min_height = f'{100 * len(items)}px'
        self._layout = VBox(**kwargs)
        for item in items:
            self._layout.children = self._layout.children + (item,)
        Accordion.__init__(self, children=[self._layout])
        self.set_title(0, name)
        self.selected_index = 0

    def _set_enabled(self, value):
        super()._set_enabled(value)
        for child in self._layout.children:
            child._set_enabled(value)


# modified from:
# https://gist.github.com/elkhadiy/284900b3ea8a13ed7b777ab93a691719
class _FilePicker(object):
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
            value=op.join(
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
                if op.isdir(op.join(self._selected_dir, el)):
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
        self._selection.value = op.join(
            self._selected_dir, self._file_selector.value
        )
        self._filename.value = self._file_selector.value

    def show(self):
        self._update_selector_options()
        self._widget.layout.display = "block"
        display(self._widget)

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
        if op.isdir(self._selection.value):
            self._selected_dir = self._selection.value
            self._file_selector.options = self._get_selector_options()
            self._file_selector.rows = min(
                len(os.listdir(self._selected_dir)), self._nb_rows)

    def _select_button_clicked(self, button):
        if self._empty_selection:
            return
        result = op.join(self._selected_dir, self._filename.value)
        if self._callback is not None:
            self._callback(result)
            # the picker is shared so only one connection is allowed at a time
            self._callback = None  # reset the callback
        self.hide()

    def _cancel_button_clicked(self, button):
        self._callback = None  # reset the callback
        self.hide()

    def _parent_button_clicked(self, button):
        self._selected_dir, _ = op.split(self._selected_dir)
        self._update_selector_options()

    def _update_path(self, change):
        self._selection.value = op.join(
            self._selected_dir, self._file_selector.value
        )
        self._filename.value = self._file_selector.value


class _FileButton(_AbstractFileButton, _Widget, Button,
                  metaclass=_BaseWidget):

    def __init__(self, callback, content_filter=None, initial_directory=None,
                 save=False, is_directory=False, icon='folder', window=None):
        _AbstractFileButton.__init__(
            callback=callback, content_filter=content_filter,
            initial_directory=initial_directory, save=save,
            is_directory=is_directory)
        _Widget.__init__(self)
        self._file_picker = _FilePicker()

        def fp_callback(x=None):
            # Note, in order to display the file picker where the button was,
            # the output must be cleared and then redrawn when finished
            if window is not None:
                clear_output()
            self._file_picker.set_directory_only(is_directory)

            def callback_with_show(name):
                window._show()
                callback(name)

            self._file_picker.connect(
                callback if window is None else callback_with_show)
            self._file_picker.show()

        Button.__init__(self, **_BASE_KWARGS)
        self.on_click(fp_callback)
        self.icon = _ICON_LUT[icon]


class _PlayMenu(_AbstractPlayMenu, _Widget, VBox, metaclass=_BaseWidget):

    def __init__(self, value, rng, callback):
        _AbstractPlayMenu.__init__(
            value=value, rng=rng, callback=callback)
        _Widget.__init__(self)
        kwargs = _BASE_KWARGS.copy()
        kwargs['layout'].align_items = 'center'
        kwargs['layout'].min_height = '100px'
        VBox.__init__(self, **kwargs)
        self._slider = IntSlider(value=value, min=rng[0], max=rng[1],
                                 readout=False, continuous_update=False)
        self._play_widget = Play(
            value=value, min=rng[0], max=rng[1], interval=250)
        self.children = (self._slider, self._play_widget)
        link((self._play_widget, 'value'), (self._slider, 'value'))
        self._slider.observe(lambda x: callback(x['new']), names='value')

    # play, pause, reset and loop require ipywidgets v8.0+ and so are
    # not currently tested, will be added upon release
    def _play(self):
        self.playing = True

    def _pause(self):
        self.playing = True

    def _reset(self):
        self.playing = True
        self.value = self.min

    def _loop(self):
        self.repeat = not self.repeat

    def _set_value(self, value):
        self._slider.value = value


class _Popup(_AbstractPopup, _Widget, VBox, metaclass=_BaseWidget):

    def __init__(self, title, text, info_text=None, callback=None,
                 icon='warning', buttons=None, window=None):
        _AbstractPopup.__init__(
            self, title=title, text=text, info_text=info_text,
            callback=callback, icon=icon, buttons=buttons, window=window)
        _Widget.__init__(self)
        VBox.__init__(self, **_BASE_KWARGS)

        if window is not None:
            clear_output()

        title_label = _Label(title)
        title_label._set_style(dict(fontsize='28'))
        text_label = _Label(text)
        text_label._set_style(dict(fontsize='18'))
        self.children = (title_label, text_label)
        if info_text:
            info_text_label = _Label(info_text)
            info_text_label._set_style(dict(fontsize='12'))
            self.children += (info_text_label,)

        self.icon = _ICON_LUT[icon]

        if buttons is None:
            buttons = ['Ok']

        hbox = HBox()
        self._buttons = dict()
        for button in buttons:

            def callback_with_show(x):
                if window is not None:
                    clear_output()
                    window._show()
                if callback:
                    callback(button)

            button_widget = Button(description=button)
            self._buttons[button] = button_widget
            button_widget.on_click(callback_with_show)
            hbox.children += (button_widget,)

        self.children += (hbox,)
        display(self)

    def _click(self, value):
        self._buttons[value].click()


class _BoxLayout(object):

    def _handle_scroll(self, scroll=None):
        kwargs = _BASE_KWARGS.copy()
        if scroll is not None:
            kwargs['layout'].width = f'{scroll[0]}px'
            kwargs['layout'].height = f'{scroll[1]}px'
            kwargs['overflow_x'] = 'scroll'
            kwargs['overflow_y'] = 'scroll'
        return kwargs

    def _add_widget(self, widget):
        # if pyvista plotter, needs to be shown
        if isinstance(widget, Plotter):
            widget = widget.show(
                jupyter_backend='ipyvtklink', return_viewer=True)
        if hasattr(widget, 'layout'):
            widget.layout.width = None  # unlock the fixed layout
            widget.layout.margin = "2px 0px 2px 0px"
            if not isinstance(widget, Play):
                widget.layout.min_width = "0px"
        self.children += (widget,)


class _HBoxLayout(_AbstractHBoxLayout, _BoxLayout, _Widget, HBox,
                  metaclass=_BaseWidget):

    def __init__(self, height=None, scroll=None):
        _Widget.__init__(self)
        _BoxLayout.__init__(self)
        _AbstractHBoxLayout.__init__(self, height=height, scroll=scroll)
        HBox.__init__(self, **self._handle_scroll(scroll=scroll))
        self._height = height

    def _add_widget(self, widget):
        _BoxLayout._add_widget(self, widget)
        if self._height is not None:
            for child in self.children:
                child.layout.height = \
                    f"{int(self._height / len(self.children))}px"

    def _add_stretch(self, amount=1):
        self.children += (self, _Label(' ' * 4),)


class _VBoxLayout(_AbstractVBoxLayout, _BoxLayout, _Widget, VBox,
                  metaclass=_BaseWidget):

    def __init__(self, width=None, scroll=None):
        _Widget.__init__(self)
        _BoxLayout.__init__(self)
        _AbstractVBoxLayout.__init__(self, width=width, scroll=scroll)
        VBox.__init__(self, **self._handle_scroll(scroll=scroll))
        self._width = width

    def _add_widget(self, widget):
        _BoxLayout._add_widget(self, widget)
        if self._width is not None:
            for child in self.children:
                child.layout.width = \
                    f"{int(self._width / len(self.children))}px"

    def _add_stretch(self, amount=1):
        self.children += (self, _Label(' ' * 4),)


class _GridLayout(_AbstractGridLayout, _Widget, GridBox,
                  metaclass=_BaseWidget):

    def __init__(self, height=None, width=None):
        _Widget.__init__(self)
        _AbstractVBoxLayout.__init__(height=height, width=width)
        GridBox.__init__(self, **_BASE_KWARGS)

    def _add_widget(self, widget, row=None, col=None):
        _BoxLayout._add_widget(self, widget)


class _Canvas(_AbstractCanvas, _Widget, HBox, metaclass=_BaseWidget):

    def __init__(self, width, height, dpi):
        import matplotlib.pyplot as plt
        _Widget.__init__(self)
        _AbstractCanvas.__init__(
            self, width=width, height=height, dpi=dpi)
        HBox.__init__(self, **_BASE_KWARGS)
        plt.ioff()
        self.fig, self.ax = plt.subplots(dpi=dpi)
        plt.ion()
        self.children = (self.fig.canvas,)

    def _set_size(self, width=None, height=None):
        if width:
            self.layout.width = width
        if height:
            self.layout.height = height


class _AppWindow(_AbstractAppWindow, _Widget, VBox, metaclass=_BaseWidget):

    def __init__(self, size=None, fullscreen=False):
        _AbstractAppWindow.__init__(self)
        _Widget.__init__(self)
        VBox.__init__(self, **_BASE_KWARGS)

    def _set_central_layout(self, central_layout):
        self.children = (central_layout,)

    def _close_connect(self, func, *, after=True):
        pass

    def _close_disconnect(self, after=True):
        pass

    def _clean(self):
        pass

    def _get_dpi(self):
        return 96

    def _get_size(self):
        # CSS objects don't have explicit widths and heights
        # https://github.com/jupyter-widgets/ipywidgets/issues/1639
        return (256, 256)

    def _get_cursor(self):
        pass

    def _set_cursor(self, cursor):
        pass

    def _new_cursor(self, name):
        pass

    def _show(self, block=False):
        display(self)

    def _close(self):
        clear_output()


class _3DRenderer(_PyVistaRenderer):
    _kind = 'notebook'

    def __init__(self, *args, **kwargs):
        kwargs['notebook'] = True
        super().__init__(*args, **kwargs)
        if 'show' in kwargs and kwargs['show']:
            self.show()

    def _update(self):
        if self.figure.display is not None:
            self.figure.display.update_canvas()

    @contextmanager
    def _ensure_minimum_sizes(self):
        yield

    def show(self):
        viewer = self.plotter.show(
            jupyter_backend="ipyvtklink", return_viewer=True)
        viewer.layout.width = None  # unlock the fixed layout
        display(viewer)


# ------------------------------------
# Non-object-based Widget Abstractions
# ------------------------------------
# These are planned to be deprecated in favor of the simpler, object-
# oriented abstractions above when time allows.

# modified from:
# https://gist.github.com/elkhadiy/284900b3ea8a13ed7b777ab93a691719
class _FilePckr:
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


class _IpyWidget(_AbstractWdgt):
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
        self._file_picker = _FilePckr(rows=10)
        kwargs["notebook"] = True
        fullscreen = kwargs.pop('fullscreen', False)
        if not _notebook_vtk_works():
            raise RuntimeError(
                'Using the notebook backend on Linux requires a compatible '
                'VTK setup. Consider using Xfvb or xvfb-run to set up a '
                'working virtual display, or install VTK with OSMesa enabled.'
            )
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
