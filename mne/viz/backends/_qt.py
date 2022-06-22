"""Qt implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
import os
import platform
import sys

import pyvista
from pyvistaqt.plotting import MainWindow
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas

from qtpy.QtCore import Qt, QLocale, QLibraryInfo, QTimer
from qtpy.QtGui import QIcon, QCursor
from qtpy.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout,
                            QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget,
                            QSizePolicy, QProgressBar, QScrollArea,
                            QLayout, QCheckBox, QButtonGroup, QRadioButton,
                            QLineEdit, QGridLayout, QFileDialog, QPushButton,
                            QMessageBox)

from ._pyvista import _PyVistaRenderer
from ._pyvista import (_close_all, _close_3d_figure, _check_3d_figure,  # noqa: F401,E501 analysis:ignore
                       _set_3d_view, _set_3d_title, _take_3d_screenshot)  # noqa: F401,E501 analysis:ignore
from ._abstract import (_AbstractWindow, _AbstractHBoxLayout,
                        _AbstractVBoxLayout, _AbstractGridLayout,
                        _AbstractWidget, _AbstractMplCanvas,
                        _AbstractDialog, _AbstractLabel, _AbstractButton,
                        _AbstractSlider, _AbstractCheckBox, _AbstractSpinBox,
                        _AbstractComboBox, _AbstractRadioButtons,
                        _AbstractGroupBox, _AbstractText, _AbstractFileButton,
                        _AbstractPlayMenu, _AbstractProgressBar)
from ._utils import (_qt_disable_paint, _qt_get_stylesheet, _qt_is_dark,
                     _qt_detect_theme, _qt_raise_window, _init_mne_qtapp)
from ...utils import get_config
from ...fixes import _compare_version

# Adapted from matplotlib
if (sys.platform == 'darwin' and
        _compare_version(platform.mac_ver()[0], '>=', '10.16') and
        QLibraryInfo.version().segments() <= [5, 15, 2]):
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


# fix for qscroll needing two layouts, one parent, one child
def _get_layout(layout):
    if hasattr(layout, '_parent_layout'):
        return layout._parent_layout
    return layout


# -------
# Widgets
# -------
# The metaclasses need to share a base class in order for the inheritance
# not to conflict, http://www.phyast.pitt.edu/~micheles/python/metatype.html
# https://stackoverflow.com/questions/28720217/multiple-inheritance-metaclass-conflict

class _BaseWidget(type(QWidget), type(_AbstractWidget)):
    pass


# The inheritance has to be in this order for the _Widget and the opposite for
# the widgets (e.g. _PushButton) that inherit from it, not sure why
class _Widget(_AbstractWidget, QWidget, metaclass=_BaseWidget):

    tooltip = None
    _to_qt = dict(
        escape=Qt.Key_Escape,
        up=Qt.Key_Up,
        down=Qt.Key_Down,
        left=Qt.Key_Left,
        right=Qt.Key_Right,
        page_up=Qt.Key_PageUp,
        page_down=Qt.Key_PageDown,
    )
    _from_qt = {v: k for k, v in _to_qt.items()}

    def __init__(self):
        _AbstractWidget.__init__()
        QWidget.__init__(self)

    def _show(self):
        self.show()

    def _hide(self):
        self.hide()

    def _set_enabled(self, state):
        self.setEnabled(state)

    def _is_enabled(self):
        return self.isEnabled()

    def _update(self, repaint=True):
        self.update()
        if repaint:
            self.repaint()

    def _get_tooltip(self):
        return self.toolTip()

    def _set_tooltip(self, tooltip):
        self.setToolTip(tooltip)

    def _set_style(self, style):
        stylesheet = ""
        for key, val in style.items():
            stylesheet = stylesheet + f"{key}:{val};"
        self.setStyleSheet(stylesheet)

    def _add_keypress(self, callback):
        self.keyPressEvent = lambda event: callback(
            self._from_qt[event.key()] if event.key() in self._from_qt else
            event.text())

    def _set_focus(self):
        self.setFocus()

    def _set_layout(self, layout):
        self.setLayout(_get_layout(layout))

    def _set_theme(self, theme=None):
        if theme is None:
            default_theme = _qt_detect_theme()
        else:
            default_theme = theme
        theme = get_config('MNE_3D_OPTION_THEME', default_theme)
        stylesheet = _qt_get_stylesheet(theme)
        self.setStyleSheet(stylesheet)
        if _qt_is_dark(self):
            QIcon.setThemeName('dark')
        else:
            QIcon.setThemeName('light')


class _Label(QLabel, _AbstractLabel, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, center=False, selectable=False):
        _AbstractLabel.__init__(value, center=center, selectable=selectable)
        _Widget.__init__(self)
        QLabel.__init__(self)
        self.setText(value)
        if center:
            self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        if selectable:
            self.setTextInteractionFlags(Qt.TextSelectableByMouse)


class _Text(QLineEdit, _AbstractText, _Widget, metaclass=_BaseWidget):

    def __init__(self, value=None, placeholder=None, callback=None):
        _AbstractText.__init__(value=value, placeholder=placeholder,
                               callback=callback)
        _Widget.__init__(self)
        QLineEdit.__init__(self, value)
        self.setPlaceholderText(placeholder)
        if callback is not None:
            self.textChanged.connect(callback)


class _Button(QPushButton, _AbstractButton, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, callback):
        _AbstractButton.__init__(value=value, callback=callback)
        _Widget.__init__(self)
        QPushButton.__init__(self)
        self.setText(value)
        self.released.connect(callback)

    def _click(self):
        self.click()

    def _set_icon(self, icon):
        self.setIcon(QIcon.fromTheme(icon))


class _Slider(QSlider, _AbstractSlider, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, rng, callback, horizontal=True):
        _AbstractSlider.__init__(value=value, rng=rng, callback=callback,
                                 horizontal=horizontal)
        _Widget.__init__(self)
        QSlider.__init__(self, Qt.Horizontal if horizontal else Qt.Vertical)
        self.setMinimum(rng[0])
        self.setMaximum(rng[1])
        self.setValue(value)
        self.valueChanged.connect(callback)

    def _set_value(self, value):
        self.setValue(value)

    def _get_value(self):
        return self.value()

    def _set_range(self, rng):
        self.setRange(int(rng[0]), int(rng[1]))


class _ProgressBar(QProgressBar, _AbstractProgressBar, _Widget,
                   metaclass=_BaseWidget):

    def __init__(self, count):
        _AbstractProgressBar.__init__(count=count)
        _Widget.__init__(self)
        QProgressBar.__init__(self)
        self.setMaximum(count)

    def _increment(self):
        self.setValue(self.value() + 1)


class _CheckBox(QCheckBox, _AbstractCheckBox, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, callback):
        _AbstractCheckBox.__init__(value=value, callback=callback)
        _Widget.__init__(self)
        QCheckBox.__init__(self)
        self.setChecked(value)
        self.stateChanged.connect(lambda x: callback(bool(x)))

    def _set_checked(self, checked):
        self.setChecked(checked)

    def _get_checked(self):
        return self.checkState() != Qt.Unchecked


class _SpinBox(QSpinBox, _AbstractSpinBox, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, rng, callback, step=None):
        _AbstractSpinBox.__init__(value=value, rng=rng, callback=callback,
                                  step=step)
        _Widget.__init__(self)
        QSpinBox.__init__(self)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimum(rng[0])
        self.setMaximum(rng[1])
        self.setKeyboardTracking(False)
        if step is None:
            inc = (rng[1] - rng[0]) / 20.
            self.setSingleStep(inc)
        else:
            self.setSingleStep(step)
        self.setValue(value)
        self.valueChanged.connect(callback)

    def _set_value(self, value):
        self.setValue(value)

    def _get_value(self):
        return self.value()


class _ComboBox(QComboBox, _AbstractComboBox, _Widget, metaclass=_BaseWidget):

    def __init__(self, value, items, callback):
        _AbstractComboBox.__init__(value=value, items=items, callback=callback)
        _Widget.__init__(self)
        QComboBox.__init__(self)
        self.addItems(items)
        self.setCurrentText(value)
        self.currentTextChanged.connect(callback)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

    def _set_value(self, value):
        self.setCurrentText(value)

    def _get_value(self):
        return self.currentText()


class _RadioButtons(QVBoxLayout, _AbstractRadioButtons, _Widget,
                    metaclass=_BaseWidget):

    def __init__(self, value, items, callback):
        _AbstractRadioButtons.__init__(
            value=value, items=items, callback=callback)
        _Widget.__init__(self)
        QVBoxLayout.__init__(self)
        self._button_group = QButtonGroup()
        self._button_group.setExclusive(True)
        for val in items:
            button = QRadioButton(val)
            if val == value:
                button.setChecked(True)
            self._button_group.addButton(button)
            self.addWidget(button)
        self._button_group.buttonClicked.connect(
            lambda button: callback(button.text()))

    def _set_value(self, value):
        self.setCurrentText(value)

    def _get_value(self):
        return self.checkedButton().text()


class _GroupBox(QGroupBox, _AbstractGroupBox, _Widget, metaclass=_BaseWidget):

    def __init__(self, name, items):
        _AbstractGroupBox.__init__(name=name, items=items)
        _Widget.__init__(self)
        QGroupBox.__init__(self, name)
        self._layout = QVBoxLayout()
        for item in items:
            self._layout.addWidget(item)
        self.setLayout(self._layout)


class _FileButton(_Button, _AbstractFileButton, _Widget,
                  metaclass=_BaseWidget):

    def __init__(self, callback, content_filter=None, initial_directory=None,
                 save=False, is_directory=False, window=None):
        _AbstractFileButton.__init__(
            callback=callback, content_filter=content_filter,
            initial_directory=initial_directory, save=save,
            is_directory=is_directory, window=window)
        _Widget.__init__(self)

        def fp_callback():
            if is_directory:
                name = QFileDialog.getExistingDirectory(
                    directory=initial_directory
                )
            elif save:
                name = QFileDialog.getSaveFileName(
                    directory=initial_directory, filter=content_filter)
            else:
                name = QFileDialog.getOpenFileName(
                    directory=initial_directory, filter=content_filter)
            name = name[0] if isinstance(name, tuple) else name
            # handle the cancel button
            if len(name) == 0:
                return
            callback(name)

        _Button.__init__(self, '', callback=fp_callback)
        self._set_icon('folder')


class _PlayMenu(QVBoxLayout, _AbstractPlayMenu, _Widget,
                metaclass=_BaseWidget):

    def __init__(self, value, rng, callback):
        _AbstractPlayMenu.__init__(
            value=value, rng=rng, callback=callback)
        _Widget.__init__(self)
        QVBoxLayout.__init__(self)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(rng[0])
        self._slider.setMaximum(rng[1])
        self._slider.setValue(value)
        self._slider.setTracking(False)
        self._slider.valueChanged.connect(callback)
        self._nav_hbox = QHBoxLayout()
        self._play = QPushButton()
        self._play.setIcon(QIcon.fromTheme('play'))
        self._nav_hbox.addWidget(self._play)
        self._pause = QPushButton()
        self._pause.setIcon(QIcon.fromTheme('pause'))
        self._nav_hbox.addWidget(self._pause)
        self._reset = QPushButton()
        self._reset.setIcon(QIcon.fromTheme('reset'))
        self._nav_hbox.addWidget(self._reset)
        self._loop = QPushButton()
        self._loop.setIcon(QIcon.fromTheme('restore'))
        self._loop.setStyleSheet('background-color : lightgray;')
        self._loop._checked = True

        def loop_callback():
            self._loop._checked = not self._loop._checked
            color = 'lightgray' if self._loop._checked else 'darkgray'
            self._loop.setStyleSheet(f'background-color : {color};')

        self._loop.released.connect(loop_callback)
        self._nav_hbox.addWidget(self._loop)
        self._timer = QTimer()

        def timer_callback():
            value = self._slider.value() + 1
            if value > rng[1]:
                if self._loop._checked:
                    self._timer.stop()
                value = rng[0]
            self._slider.setValue(value)

        self._timer.timeout.connect(timer_callback)
        self._timer.setInterval(250)
        self._play.released.connect(self._timer.start)
        self._pause.released.connect(self._timer.stop)
        self._reset.released.connect(lambda: self._slider.setValue(rng[0]))
        self.addWidget(self._slider)
        self.addLayout(self._nav_hbox)


class _Dialog(QMessageBox, _AbstractDialog, _Widget, metaclass=_BaseWidget):

    def __init__(self, title, text, info_text, callback,
                 icon='warning', buttons=None, window=None):
        _AbstractDialog.__init__(
            self, title=title, text=text, info_text=info_text,
            callback=callback, icon=icon, buttons=buttons, window=window)
        _Widget.__init__(self)
        QMessageBox.__init__(self)
        self.setWindowTitle(title)
        self.setText(text)
        # icon is one of _Dialog.supported_icon_names
        if icon is not None:
            self.setIcon(getattr(QMessageBox, icon.title()))
        self.setInformativeText(info_text)

        if buttons is None:
            buttons = ['Ok']

        button_ids = list()
        for button in buttons:
            # button is one of _Dialog.supported_button_names
            button_id = getattr(QMessageBox, button)
            button_ids.append(button_id)
        standard_buttons = default_button = button_ids[0]
        for button_id in button_ids[1:]:
            standard_buttons |= button_id
        self.setStandardButtons(standard_buttons)
        self.setDefaultButton(default_button)
        self.buttonClicked.connect(lambda button: callback(button.text()))
        self._show()


class _ScrollArea(QScrollArea):

    def __init__(self, width, height, widget):
        QScrollArea.__init__(self)
        self.setWidget(widget)
        self.setFixedSize(width, height)
        self.setWidgetResizable(True)


class _HBoxLayout(QHBoxLayout, _AbstractHBoxLayout, _Widget,
                  metaclass=_BaseWidget):

    def __init__(self, height=None, scroll=None):
        _AbstractHBoxLayout.__init__(self, height=height, scroll=scroll)
        _Widget.__init__(self)
        QHBoxLayout.__init__(self)

        if scroll is not None:
            self._scroll_widget = QWidget()
            self._parent_layout = QHBoxLayout()
            self._parent_layout.addWidget(
                _ScrollArea(scroll[0], scroll[1], self._scroll_widget))
            self._scroll_widget.setLayout(self)

        if height is not None:
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)

    def _add_widget(self, widget):
        """Add a widget to an existing layout."""
        if isinstance(widget, QLayout):
            self.addLayout(widget)
        else:
            self.addWidget(widget)

    def _add_stretch(self, amount=1):
        self.addStretch(amount)


class _VBoxLayout(QVBoxLayout, _AbstractVBoxLayout, _Widget,
                  metaclass=_BaseWidget):

    def __init__(self, width=None, scroll=None):
        _AbstractVBoxLayout.__init__(self, width=width, scroll=scroll)
        _Widget.__init__(self)
        QVBoxLayout.__init__(self)

        if scroll is not None:
            self._scroll_widget = QWidget()
            self._parent_layout = QHBoxLayout()
            self._parent_layout.addWidget(
                _ScrollArea(scroll[0], scroll[1], self._scroll_widget))
            self._scroll_widget.setLayout(self)

        if width is not None:
            self.setMinimumWidth(width)
            self.setMaximumWidth(width)

    def _add_widget(self, widget):
        """Add a widget to an existing layout."""
        if isinstance(widget, QLayout):
            self.addLayout(widget)
        else:
            self.addWidget(widget)

    def _add_stretch(self, amount=1):
        self.addStretch(amount)


class _GridLayout(QGridLayout, _AbstractGridLayout, _Widget,
                  metaclass=_BaseWidget):

    def __init__(self, height=None, width=None):
        _AbstractGridLayout.__init__(self)
        _Widget.__init__(self)
        QGridLayout.__init__(self)
        if height:
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
        if width:
            self.setMinimumWidth(width)
            self.setMaximumWidth(width)

    def _add_widget(self, widget, row=None, col=None):
        """Add a widget to an existing layout."""
        if isinstance(widget, QLayout):
            self.addLayout(widget, row, col)
        else:
            self.addWidget(widget, row, col)


class _BaseCanvas(type(FigureCanvas), type(_AbstractMplCanvas)):
    pass


class _MplCanvas(FigureCanvas, _AbstractMplCanvas, metaclass=_BaseCanvas):

    def __init__(self, width, height, dpi):
        _AbstractMplCanvas.__init__(
            self, width=width, height=height, dpi=dpi)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(width)
        self.setMinimumHeight(height)


# %%
# Windows
# -------

# In theory we should be able to do this later (e.g., in _pyvista.py when
# initializing), but at least on Qt6 this has to be done earlier. So let's do
# it immediately upon instantiation of the QMainWindow class.
# TODO: This should eventually allow us to handle
# https://github.com/mne-tools/mne-python/issues/9182


class _MNEMainWindow(MainWindow):
    def __init__(self, parent=None, title=None, size=None):
        super().__init__(parent, title, size)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)


class _Window(_AbstractWindow, _MNEMainWindow, _Widget, metaclass=_BaseWidget):

    def __init__(self, size=None, fullscreen=False):
        self._app = _init_mne_qtapp()
        _AbstractWindow.__init__(self)
        _Widget.__init__(self)
        _MNEMainWindow.__init__(self, size=size)

        if fullscreen:
            self.setWindowState(Qt.WindowFullScreen)

        self._set_theme()
        self.setLocale(QLocale(QLocale.Language.English))
        self.signal_close.connect(self._clean)
        self._before_close_callbacks = list()
        self._after_close_callbacks = list()

        # patch closeEvent
        def closeEvent(event):
            # functions to call before closing
            accept_close_event = True
            for callback in self._before_close_callbacks:
                ret = callback()
                # check if one of the callbacks ignores the close event
                if isinstance(ret, bool) and not ret:
                    accept_close_event = False

            if accept_close_event:
                self.signal_close.emit()
                self._clean()
                event.accept()
            else:
                event.ignore()

            # functions to call after closing
            for callback in self._after_close_callbacks:
                callback()
        self.closeEvent = closeEvent

    def _set_central_layout(self, central_layout):
        central_widget = QWidget()
        central_widget.setLayout(_get_layout(central_layout))
        self.setCentralWidget(central_widget)

    def _get_dpi(self):
        return self.windowHandle().screen().logicalDotsPerInch()

    def _get_size(self):
        return (self.width(), self.height())

    def _get_cursor(self):
        return self.cursor()

    def _set_cursor(self, cursor):
        self.setCursor(cursor)

    def _new_cursor(self, name):
        return QCursor(getattr(Qt, name))

    def _close_connect(self, callback, *, after=True):
        if after:
            self._after_close_callbacks.append(callback)
        else:
            self._before_close_callbacks.append(callback)

    def _close_disconnect(self, after=True):
        if after:
            self._after_close_callbacks.clear()
        else:
            self._before_close_callbacks.clear()

    def _clean(self):
        self._app = None

    def _show(self):
        _qt_raise_window(self)
        _Widget._show(self)


class _Renderer(_PyVistaRenderer):
    _kind = 'qt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show(self):
        super().show()
        with _qt_disable_paint(self.plotter):
            self.plotter.app_window.show()
        self._update()
        for plotter in self._all_plotters:
            plotter.updateGeometry()
            plotter._render()
        # Ideally we would just put a `splash.finish(plotter.window())` in the
        # same place that we initialize this (_init_qt_app call). However,
        # the window show event is triggered (closing the splash screen) well
        # before the window actually appears for complex scenes like the coreg
        # GUI. Therefore, we close after all these events have been processed
        # here.
        self._process_events()
        splash = getattr(self.figure, 'splash', False)
        if splash:
            splash.close()
        _qt_raise_window(self.plotter.app_window)

    def _clean(self):
        self.figure._plotter = None
        self._interactor = None


@contextmanager
def _testing_context(interactive):
    from . import renderer
    orig_offscreen = pyvista.OFF_SCREEN
    orig_testing = renderer.MNE_3D_BACKEND_TESTING
    orig_interactive = renderer.MNE_3D_BACKEND_INTERACTIVE
    renderer.MNE_3D_BACKEND_TESTING = True
    if interactive:
        pyvista.OFF_SCREEN = False
        renderer.MNE_3D_BACKEND_INTERACTIVE = True
    else:
        pyvista.OFF_SCREEN = True
        renderer.MNE_3D_BACKEND_INTERACTIVE = False
    try:
        yield
    finally:
        pyvista.OFF_SCREEN = orig_offscreen
        renderer.MNE_3D_BACKEND_TESTING = orig_testing
        renderer.MNE_3D_BACKEND_INTERACTIVE = orig_interactive
