"""Notebook implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import os
import os.path as op
from contextlib import contextmanager, nullcontext

from IPython.display import display, clear_output
from ipywidgets import (Widget, HBox, VBox, Button, Dropdown, IntSlider,
                        IntText, Text, IntProgress, Play, Label, HTML,
                        Checkbox, RadioButtons, Accordion, link,
                        Layout, Select, GridBox)
from ipyevents import Event

from ._abstract import (_AbstractWindow, _AbstractHBoxLayout,
                        _AbstractVBoxLayout, _AbstractGridLayout,
                        _AbstractWidget, _AbstractMplCanvas,
                        _AbstractDialog, _AbstractLabel,
                        _AbstractButton, _AbstractSlider,
                        _AbstractCheckBox, _AbstractSpinBox, _AbstractComboBox,
                        _AbstractRadioButtons, _AbstractGroupBox,
                        _AbstractText, _AbstractFileButton,
                        _AbstractPlayMenu, _AbstractProgressBar)
from ._pyvista import _PyVistaRenderer, Plotter   # noqa: F401,E501, analysis:ignore


# from: https://fontawesome.com/icons
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

        def fp_callback():
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
        self._play = Play(value=value, min=rng[0], max=rng[1], interval=250)
        self.children = (self._slider, self._play)
        link((self._play, 'value'), (self._slider, 'value'))
        self._slider.observe(lambda x: callback(x['new']), names='value')


class _Dialog(_AbstractDialog, _Widget, VBox, metaclass=_BaseWidget):

    def __init__(self, title, text, info_text=None, callback=None,
                 icon='warning', buttons=None, window=None):
        _AbstractDialog.__init__(
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

        self.icon = _ICON_LUT(icon)

        if buttons is None:
            buttons = ['Ok']

        hbox = HBox()
        for button in buttons:

            def callback_with_show(x):
                if window is not None:
                    clear_output()
                    window._show()
                if callback:
                    callback(button)

            button_widget = Button(description=button)
            button_widget.on_click(callback_with_show)
            hbox.children += (button_widget,)

        self.children += (hbox,)
        display(self)


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
                jupyter_backend="ipyvtklink", return_viewer=True)
            widget.layout.width = None  # unlock the fixed layout
        if hasattr(widget, 'layout'):
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


class _MplCanvas(_AbstractMplCanvas, _Widget, HBox, metaclass=_BaseWidget):

    def __init__(self, width, height, dpi):
        import matplotlib.pyplot as plt
        _Widget.__init__(self)
        _AbstractMplCanvas.__init__(
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


class _Window(_AbstractWindow, _Widget, VBox, metaclass=_BaseWidget):

    def __init__(self, size=None, fullscreen=False):
        _AbstractWindow.__init__(self)
        _Widget.__init__(self)
        VBox.__init__(self, **_BASE_KWARGS)

    def _set_central_layout(self, central_layout):
        self.children = (central_layout,)

    def _close_connect(self, func, *, after=True):
        pass

    def _close_disconnect(self, after=True):
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


class _Renderer(_PyVistaRenderer):
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


# self._dock_width = 302


_testing_context = nullcontext
