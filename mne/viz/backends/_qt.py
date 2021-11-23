"""Qt implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager

import pyvista
from pyvistaqt.plotting import FileDialog

from PyQt5.QtCore import Qt, pyqtSignal, QLocale
from PyQt5.QtGui import QIcon, QImage, QPixmap, QCursor
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QLabel, QToolButton, QMenuBar,
                             QSlider, QSpinBox, QVBoxLayout, QWidget,
                             QSizePolicy, QScrollArea, QStyle, QProgressBar,
                             QStyleOptionSlider, QLayout, QCheckBox,
                             QButtonGroup, QRadioButton, QLineEdit,
                             QFileDialog)

from ._pyvista import _PyVistaRenderer
from ._pyvista import (_close_all, _close_3d_figure, _check_3d_figure,  # noqa: F401,E501 analysis:ignore
                       _set_3d_view, _set_3d_title, _take_3d_screenshot)  # noqa: F401,E501 analysis:ignore
from ._abstract import (_AbstractDock, _AbstractToolBar, _AbstractMenuBar,
                        _AbstractStatusBar, _AbstractLayout, _AbstractWidget,
                        _AbstractWindow, _AbstractMplCanvas, _AbstractPlayback,
                        _AbstractBrainMplCanvas, _AbstractMplInterface,
                        _AbstractWidgetList)
from ._utils import _init_qt_resources, _qt_disable_paint
from ..utils import logger


class _QtLayout(_AbstractLayout):
    def _layout_initialize(self, max_width):
        pass

    def _layout_add_widget(self, layout, widget, stretch=0):
        if isinstance(widget, QLayout):
            layout.addLayout(widget)
        else:
            layout.addWidget(widget, stretch)


class _QtDock(_AbstractDock, _QtLayout):
    def _dock_initialize(self, window=None, name="Controls",
                         area="left"):
        window = self._window if window is None else window
        qt_area = Qt.LeftDockWidgetArea if area == "left" \
            else Qt.RightDockWidgetArea
        self._dock, self._dock_layout = _create_dock_widget(
            self._window, name, qt_area)
        if area == "left":
            window.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        else:
            window.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

    def _dock_finalize(self):
        self._dock.setMinimumSize(self._dock.sizeHint().width(), 0)
        self._dock_add_stretch(self._dock_layout)

    def _dock_show(self):
        self._dock.show()

    def _dock_hide(self):
        self._dock.hide()

    def _dock_add_stretch(self, layout=None):
        layout = self._dock_layout if layout is None else layout
        layout.addStretch()

    def _dock_add_layout(self, vertical=True):
        layout = QVBoxLayout() if vertical else QHBoxLayout()
        return layout

    def _dock_add_label(self, value, align=False, layout=None):
        layout = self._dock_layout if layout is None else layout
        widget = QLabel()
        if align:
            widget.setAlignment(Qt.AlignCenter)
        widget.setText(value)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_button(self, name, callback, layout=None):
        layout = self._dock_layout if layout is None else layout
        # If we want one with text instead of an icon, we should use
        # QPushButton(name)
        widget = QToolButton()
        widget.clicked.connect(callback)
        widget.setText(name)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_named_layout(self, name, layout=None, compact=True):
        layout = self._dock_layout if layout is None else layout
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
        slider_class = QFloatSlider if double else QSlider
        cast = float if double else int
        widget = slider_class(Qt.Horizontal)
        widget.setMinimum(cast(rng[0]))
        widget.setMaximum(cast(rng[1]))
        widget.setValue(cast(value))
        widget.valueChanged.connect(callback)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_check_box(self, name, value, callback, layout=None):
        layout = self._dock_layout if layout is None else layout
        widget = QCheckBox(name)
        widget.setChecked(value)
        widget.stateChanged.connect(callback)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_spin_box(self, name, value, rng, callback,
                           compact=True, double=True, step=None,
                           layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        value = value if double else int(value)
        widget = QDoubleSpinBox() if double else QSpinBox()
        widget.setAlignment(Qt.AlignCenter)
        widget.setMinimum(rng[0])
        widget.setMaximum(rng[1])
        widget.setKeyboardTracking(False)
        if step is None:
            inc = (rng[1] - rng[0]) / 20.
            inc = max(int(round(inc)), 1) if not double else inc
            widget.setSingleStep(inc)
        else:
            widget.setSingleStep(step)
        widget.setValue(value)
        widget.valueChanged.connect(callback)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        widget = QComboBox()
        widget.addItems(rng)
        widget.setCurrentText(value)
        widget.currentTextChanged.connect(callback)
        widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_radio_buttons(self, value, rng, callback, vertical=True,
                                layout=None):
        layout = self._dock_layout if layout is None else layout
        group_layout = QVBoxLayout() if vertical else QHBoxLayout()
        group = QButtonGroup()
        for val in rng:
            button = QRadioButton(val)
            if val == value:
                button.setChecked(True)
            group.addButton(button)
            self._layout_add_widget(group_layout, button)

        def func(button):
            callback(button.text())
        group.buttonClicked.connect(func)
        self._layout_add_widget(layout, group_layout)
        return _QtWidgetList(group)

    def _dock_add_group_box(self, name, layout=None):
        layout = self._dock_layout if layout is None else layout
        hlayout = QVBoxLayout()
        widget = QGroupBox(name)
        widget.setLayout(hlayout)
        self._layout_add_widget(layout, widget)
        return hlayout

    def _dock_add_text(self, name, value, placeholder, layout=None):
        layout = self._dock_layout if layout is None else layout
        widget = QLineEdit(value)
        widget.setPlaceholderText(placeholder)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_file_button(self, name, desc, func, value=None, save=False,
                              directory=False, input_text_widget=True,
                              placeholder="Type a file name", layout=None):
        layout = self._dock_layout if layout is None else layout
        if input_text_widget:
            hlayout = self._dock_add_layout(vertical=False)
            text_widget = self._dock_add_text(
                name=f"{name}_field",
                value=value,
                placeholder=placeholder,
                layout=hlayout,
            )

            def sync_text_widget(s):
                text_widget.set_value(s)
        else:
            hlayout = layout

        def callback():
            if directory:
                name = QFileDialog.getExistingDirectory()
            elif save:
                name = QFileDialog.getSaveFileName()
            else:
                name = QFileDialog.getOpenFileName()
            name = name[0] if isinstance(name, tuple) else name
            # handle the cancel button
            if len(name) == 0:
                return
            if input_text_widget:
                sync_text_widget(name)
            func(name)

        button_widget = self._dock_add_button(
            name=desc,
            callback=callback,
            layout=hlayout,
        )
        if input_text_widget:
            self._layout_add_widget(layout, hlayout)
            return _QtWidgetList([text_widget, button_widget])
        else:
            return _QtWidget(button_widget)


class QFloatSlider(QSlider):
    """Slider that handles float values."""

    valueChanged = pyqtSignal(float)

    def __init__(self, ori, parent=None):
        """Initialize the slider."""
        super().__init__(ori, parent)
        self._opt = QStyleOptionSlider()
        self.initStyleOption(self._opt)
        self._gr = self.style().subControlRect(
            QStyle.CC_Slider, self._opt, QStyle.SC_SliderGroove, self)
        self._sr = self.style().subControlRect(
            QStyle.CC_Slider, self._opt, QStyle.SC_SliderHandle, self)
        self._precision = 10000
        super().valueChanged.connect(self._convert)

    def _convert(self, value):
        self.valueChanged.emit(value / self._precision)

    def minimum(self):
        """Get the minimum."""
        return super().minimum() / self._precision

    def setMinimum(self, value):
        """Set the minimum."""
        super().setMinimum(int(value * self._precision))

    def maximum(self):
        """Get the maximum."""
        return super().maximum() / self._precision

    def setMaximum(self, value):
        """Set the maximum."""
        super().setMaximum(int(value * self._precision))

    def value(self):
        """Get the current value."""
        return super().value() / self._precision

    def setValue(self, value):
        """Set the current value."""
        super().setValue(int(value * self._precision))

    # Adapted from:
    # https://stackoverflow.com/questions/52689047/moving-qslider-to-mouse-click-position  # noqa: E501
    def mousePressEvent(self, event):
        """Add snap-to-location handling."""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        sr = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        if (event.button() != Qt.LeftButton or sr.contains(event.pos())):
            super().mousePressEvent(event)
            return
        if self.orientation() == Qt.Vertical:
            half = (0.5 * sr.height()) + 0.5
            max_ = self.height()
            pos = max_ - event.y()
        else:
            half = (0.5 * sr.width()) + 0.5
            max_ = self.width()
            pos = event.x()
        max_ = max_ - 2 * half
        pos = min(max(pos - half, 0), max_) / max_
        val = self.minimum() + (self.maximum() - self.minimum()) * pos
        val = (self.maximum() - val) if self.invertedAppearance() else val
        self.setValue(val)
        event.accept()
        # Process afterward so it's seen as a drag
        super().mousePressEvent(event)


class _QtToolBar(_AbstractToolBar, _QtLayout):
    def _tool_bar_load_icons(self):
        _init_qt_resources()
        self.icons = dict()
        self.icons["help"] = QIcon(":/help.svg")
        self.icons["play"] = QIcon(":/play.svg")
        self.icons["pause"] = QIcon(":/pause.svg")
        self.icons["reset"] = QIcon(":/reset.svg")
        self.icons["scale"] = QIcon(":/scale.svg")
        self.icons["clear"] = QIcon(":/clear.svg")
        self.icons["movie"] = QIcon(":/movie.svg")
        self.icons["restore"] = QIcon(":/restore.svg")
        self.icons["screenshot"] = QIcon(":/screenshot.svg")
        self.icons["visibility_on"] = QIcon(":/visibility_on.svg")
        self.icons["visibility_off"] = QIcon(":/visibility_off.svg")

    def _tool_bar_initialize(self, name="default", window=None):
        self.actions = dict()
        window = self._window if window is None else window
        self._tool_bar = window.addToolBar(name)
        self._tool_bar_layout = self._tool_bar.layout()

    def _tool_bar_add_button(self, name, desc, func, icon_name=None,
                             shortcut=None):
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        self.actions[name] = self._tool_bar.addAction(icon, desc, func)
        if shortcut is not None:
            self.actions[name].setShortcut(shortcut)

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].setIcon(self.icons[icon_name])

    def _tool_bar_add_text(self, name, value, placeholder):
        pass

    def _tool_bar_add_spacer(self):
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._tool_bar.addWidget(spacer)

    def _tool_bar_add_file_button(self, name, desc, func, shortcut=None):
        def callback():
            return FileDialog(
                self.plotter.app_window,
                callback=func,
            )

        self._tool_bar_add_button(
            name=name,
            desc=desc,
            func=callback,
            shortcut=shortcut,
        )

    def _tool_bar_add_play_button(self, name, desc, func, shortcut=None):
        self._tool_bar_add_button(name, desc, func, None, shortcut)

    def _tool_bar_set_theme(self, theme):
        if theme == 'auto':
            theme = _detect_theme()

        if theme == 'dark':
            for icon_key in self.icons:
                icon = self.icons[icon_key]
                image = icon.pixmap(80).toImage()
                image.invertPixels(mode=QImage.InvertRgb)
                self.icons[icon_key] = QIcon(QPixmap.fromImage(image))


class _QtMenuBar(_AbstractMenuBar):
    def _menu_initialize(self, window=None):
        self._menus = dict()
        self._menu_actions = dict()
        self._menu_bar = QMenuBar()
        self._menu_bar.setNativeMenuBar(False)
        window = self._window if window is None else window
        window.setMenuBar(self._menu_bar)

    def _menu_add_submenu(self, name, desc):
        self._menus[name] = self._menu_bar.addMenu(desc)

    def _menu_add_button(self, menu_name, name, desc, func):
        menu = self._menus[menu_name]
        self._menu_actions[name] = menu.addAction(desc, func)


class _QtStatusBar(_AbstractStatusBar, _QtLayout):
    def _status_bar_initialize(self, window=None):
        window = self._window if window is None else window
        self._status_bar = window.statusBar()
        self._status_bar_layout = self._status_bar.layout()

    def _status_bar_add_label(self, value, stretch=0):
        widget = QLabel(value)
        self._layout_add_widget(self._status_bar_layout, widget, stretch)
        return _QtWidget(widget)

    def _status_bar_add_progress_bar(self, stretch=0):
        widget = QProgressBar()
        self._layout_add_widget(self._status_bar_layout, widget, stretch)
        return _QtWidget(widget)

    def _status_bar_update(self):
        self._status_bar_layout.update()


class _QtPlayback(_AbstractPlayback):
    def _playback_initialize(self, func, timeout, value, rng,
                             time_widget, play_widget):
        self.figure.plotter.add_callback(func, timeout)


class _QtMplInterface(_AbstractMplInterface):
    def _mpl_initialize(self):
        from PyQt5 import QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        self.canvas = FigureCanvasQTAgg(self.fig)
        FigureCanvasQTAgg.setSizePolicy(
            self.canvas,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self.canvas)


class _QtMplCanvas(_AbstractMplCanvas, _QtMplInterface):
    def __init__(self, width, height, dpi):
        super().__init__(width, height, dpi)
        self._mpl_initialize()


class _QtBrainMplCanvas(_AbstractBrainMplCanvas, _QtMplInterface):
    def __init__(self, brain, width, height, dpi):
        super().__init__(brain, width, height, dpi)
        self._mpl_initialize()
        if brain.separate_canvas:
            self.canvas.setParent(None)
        else:
            self.canvas.setParent(brain._renderer._window)
        self._connect()


class _QtWindow(_AbstractWindow):
    def _window_initialize(self):
        super()._window_initialize()
        self._interactor = self.figure.plotter.interactor
        self._window = self.figure.plotter.app_window
        self._window.setLocale(QLocale(QLocale.Language.English))
        self._window.signal_close.connect(self._window_clean)

    def _window_clean(self):
        self.figure.plotter = None
        self._interactor = None

    def _window_close_connect(self, func):
        self._window.signal_close.connect(func)

    def _window_get_dpi(self):
        return self._window.windowHandle().screen().logicalDotsPerInch()

    def _window_get_size(self):
        w = self._interactor.geometry().width()
        h = self._interactor.geometry().height()
        return (w, h)

    def _window_get_simple_canvas(self, width, height, dpi):
        return _QtMplCanvas(width, height, dpi)

    def _window_get_mplcanvas(self, brain, interactor_fraction, show_traces,
                              separate_canvas):
        w, h = self._window_get_mplcanvas_size(interactor_fraction)
        self._interactor_fraction = interactor_fraction
        self._show_traces = show_traces
        self._separate_canvas = separate_canvas
        self._mplcanvas = _QtBrainMplCanvas(
            brain, w, h, self._window_get_dpi())
        return self._mplcanvas

    def _window_adjust_mplcanvas_layout(self):
        canvas = self._mplcanvas.canvas
        self._mpl_dock, dock_layout = _create_dock_widget(
            self._window, "Traces", Qt.BottomDockWidgetArea)
        dock_layout.addWidget(canvas)

    def _window_get_cursor(self):
        return self._interactor.cursor()

    def _window_set_cursor(self, cursor):
        self._interactor.setCursor(cursor)

    def _window_new_cursor(self, name):
        return QCursor(getattr(Qt, name))

    @contextmanager
    def _window_ensure_minimum_sizes(self):
        sz = self.figure.store['window_size']
        adjust_mpl = (self._show_traces and not self._separate_canvas)
        # plotter:            pyvista.plotting.qt_plotting.BackgroundPlotter
        # plotter.interactor: vtk.qt.QVTKRenderWindowInteractor.QVTKRenderWindowInteractor -> QWidget  # noqa
        # plotter.app_window: pyvista.plotting.qt_plotting.MainWindow -> QMainWindow  # noqa
        # plotter.frame:      QFrame with QVBoxLayout with plotter.interactor as centralWidget  # noqa
        # plotter.ren_win:    vtkXOpenGLRenderWindow
        self._interactor.setMinimumSize(*sz)
        # Lines like this are useful for debugging these issues:
        # print('*' * 80)
        # print(0, self._interactor.app_window.size().height(), self._interactor.size().height(), self._mpl_dock.widget().height(), self._mplcanvas.canvas.size().height())  # noqa
        if adjust_mpl:
            mpl_h = int(round((sz[1] * self._interactor_fraction) /
                              (1 - self._interactor_fraction)))
            self._mplcanvas.canvas.setMinimumSize(sz[0], mpl_h)
            self._mpl_dock.widget().setMinimumSize(sz[0], mpl_h)
        try:
            yield  # show
        finally:
            # 1. Process events
            self._process_events()
            self._process_events()
            # 2. Get the window and interactor sizes that work
            win_sz = self._window.size()
            ren_sz = self._interactor.size()
            # 3. Undo the min size setting and process events
            self._interactor.setMinimumSize(0, 0)
            if adjust_mpl:
                self._mplcanvas.canvas.setMinimumSize(0, 0)
                self._mpl_dock.widget().setMinimumSize(0, 0)
            self._process_events()
            self._process_events()
            # 4. Compute the extra height required for dock decorations and add
            win_h = win_sz.height()
            if adjust_mpl:
                win_h += max(
                    self._mpl_dock.widget().size().height() - mpl_h, 0)
            # 5. Resize the window and interactor to the correct size
            #    (not sure why, but this is required on macOS at least)
            self._interactor.window_size = (win_sz.width(), win_h)
            self._interactor.resize(ren_sz.width(), ren_sz.height())
            self._process_events()
            self._process_events()

    def _window_set_theme(self, theme):
        if theme == 'auto':
            theme = _detect_theme()

        if theme == 'dark':
            try:
                import qdarkstyle
            except ModuleNotFoundError:
                logger.info('For Dark-Mode "qdarkstyle" has to be installed! '
                            'You can install it with `pip install qdarkstyle`')
                stylesheet = None
            else:
                stylesheet = qdarkstyle.load_stylesheet()
        elif theme != 'light':
            with open(theme, 'r') as file:
                stylesheet = file.read()
        else:
            stylesheet = None

        self._window.setStyleSheet(stylesheet)


class _QtWidgetList(_AbstractWidgetList):
    def __init__(self, src):
        self._src = src
        self._widgets = list()
        if isinstance(self._src, QButtonGroup):
            widgets = self._src.buttons()
        else:
            widgets = src
        for widget in widgets:
            if not isinstance(widget, _QtWidget):
                widget = _QtWidget(widget)
            self._widgets.append(widget)

    def set_enabled(self, state):
        for widget in self._widgets:
            widget.set_enabled(state)

    def get_value(self, idx):
        return self._widgets[idx].get_value()

    def set_value(self, idx, value):
        if isinstance(self._src, QButtonGroup):
            self._widgets[idx].set_value(True)
        else:
            self._widgets[idx].set_value(value)


class _QtWidget(_AbstractWidget):
    def set_value(self, value):
        if isinstance(self._widget, (QRadioButton, QToolButton)):
            self._widget.click()
        else:
            if hasattr(self._widget, "setValue"):
                self._widget.setValue(value)
            elif hasattr(self._widget, "setCurrentText"):
                self._widget.setCurrentText(value)
            elif hasattr(self._widget, "setChecked"):
                self._widget.setChecked(value)
            else:
                assert hasattr(self._widget, "setText")
                self._widget.setText(value)

    def get_value(self):
        if hasattr(self._widget, "value"):
            return self._widget.value()
        elif hasattr(self._widget, "currentText"):
            return self._widget.currentText()
        elif hasattr(self._widget, "checkState"):
            return bool(self._widget.checkState())
        else:
            assert hasattr(self._widget, "text")
            return self._widget.text()

    def set_range(self, rng):
        self._widget.setRange(rng[0], rng[1])

    def show(self):
        self._widget.show()

    def hide(self):
        self._widget.hide()

    def set_enabled(self, state):
        self._widget.setEnabled(state)

    def update(self, repaint=True):
        self._widget.update()
        if repaint:
            self._widget.repaint()


class _Renderer(_PyVistaRenderer, _QtDock, _QtToolBar, _QtMenuBar,
                _QtStatusBar, _QtWindow, _QtPlayback):
    _kind = 'qt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_initialize()

    def show(self):
        super().show()
        with _qt_disable_paint(self.plotter):
            with self._window_ensure_minimum_sizes():
                self.plotter.app_window.show()
        self._update()
        for plotter in self._all_plotters:
            plotter.updateGeometry()
            plotter._render()
        self._process_events()


def _create_dock_widget(window, name, area):
    # create dock widget
    dock = QDockWidget(name)
    # add scroll area
    scroll = QScrollArea(dock)
    dock.setWidget(scroll)
    # give the scroll area a child widget
    widget = QWidget(scroll)
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    dock.setAllowedAreas(area)
    dock.setTitleBarWidget(QLabel(name))
    window.addDockWidget(area, dock)
    dock_layout = QVBoxLayout()
    widget.setLayout(dock_layout)
    # Fix resize grip size
    # https://stackoverflow.com/a/65050468/2175965
    dock.setStyleSheet("QDockWidget { margin: 4px; }")
    return dock, dock_layout


def _detect_theme():
    try:
        import darkdetect
        return darkdetect.theme().lower()
    except Exception:
        return 'light'


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
