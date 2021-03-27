"""Qt implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager
from functools import partial

import pyvista

from PyQt5.QtCore import Qt, pyqtSignal, QLocale
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QLabel, QToolButton, QMenuBar,
                             QSlider, QSpinBox, QVBoxLayout, QWidget,
                             QSizePolicy, QScrollArea, QStyle, QProgressBar,
                             QStyleOptionSlider, QLayout)

from ._pyvista import _PyVistaRenderer
from ._pyvista import (_close_all, _close_3d_figure, _check_3d_figure,  # noqa: F401,E501 analysis:ignore
                       _set_3d_view, _set_3d_title, _take_3d_screenshot)  # noqa: F401,E501 analysis:ignore
from ._abstract import (_AbstractDock, _AbstractToolBar, _AbstractMenuBar,
                        _AbstractStatusBar, _AbstractLayout, _AbstractWidget,
                        _AbstractWindow, _AbstractMplCanvas, _AbstractPlayback,
                        _AbstractBrainMplCanvas, _AbstractMplInterface)
from ._utils import _init_qt_resources, _qt_disable_paint
from ..utils import _save_ndarray_img, logger


class _QtLayout(_AbstractLayout):
    def _layout_initialize(self, max_width):
        pass

    def _layout_add_widget(self, layout, widget, max_width=None):
        if isinstance(widget, QLayout):
            layout.addLayout(widget)
        else:
            layout.addWidget(widget)


class _QtDock(_AbstractDock, _QtLayout):
    def _dock_initialize(self, window=None):
        window = self._window if window is None else window
        self.dock, self.dock_layout = _create_dock_widget(
            self._window, "Controls", Qt.LeftDockWidgetArea)
        window.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

    def _dock_finalize(self):
        self.dock.setMinimumSize(self.dock.sizeHint().width(), 0)
        self._dock_add_stretch(self.dock_layout)

    def _dock_show(self):
        self.dock.show()

    def _dock_hide(self):
        self.dock.hide()

    def _dock_add_stretch(self, layout):
        layout.addStretch()

    def _dock_add_layout(self, vertical=True):
        layout = QVBoxLayout() if vertical else QHBoxLayout()
        return layout

    def _dock_add_label(self, value, align=False, layout=None):
        layout = self.dock_layout if layout is None else layout
        widget = QLabel()
        if align:
            widget.setAlignment(Qt.AlignCenter)
        widget.setText(value)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

    def _dock_add_button(self, name, callback, layout=None):
        layout = self.dock_layout if layout is None else layout
        # If we want one with text instead of an icon, we should use
        # QPushButton(name)
        widget = QToolButton()
        widget.clicked.connect(callback)
        widget.setText(name)
        self._layout_add_widget(layout, widget)
        return _QtWidget(widget)

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
                         compact=True, double=False, layout=None,
                         stretch=0):
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

    def _dock_add_spin_box(self, name, value, rng, callback,
                           compact=True, double=True, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        value = value if double else int(value)
        widget = QDoubleSpinBox() if double else QSpinBox()
        widget.setAlignment(Qt.AlignCenter)
        widget.setMinimum(rng[0])
        widget.setMaximum(rng[1])
        inc = (rng[1] - rng[0]) / 20.
        inc = max(int(round(inc)), 1) if not double else inc
        widget.setKeyboardTracking(False)
        widget.setSingleStep(inc)
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

    def _dock_add_group_box(self, name, layout=None):
        layout = self.dock_layout if layout is None else layout
        hlayout = QVBoxLayout()
        widget = QGroupBox(name)
        widget.setLayout(hlayout)
        self._layout_add_widget(layout, widget)
        return hlayout


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
        self.tool_bar = window.addToolBar(name)

    def _tool_bar_add_button(self, name, desc, func, icon_name=None,
                             shortcut=None):
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        self.actions[name] = self.tool_bar.addAction(icon, desc, func)
        if shortcut is not None:
            self.actions[name].setShortcut(shortcut)

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].setIcon(self.icons[icon_name])

    def _tool_bar_add_text(self, name, value, placeholder):
        pass

    def _tool_bar_add_spacer(self):
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tool_bar.addWidget(spacer)

    def _tool_bar_add_screenshot_button(self, name, desc, func):
        def _screenshot():
            img = func()
            try:
                from pyvista.plotting.qt_plotting import FileDialog
            except ImportError:
                from pyvistaqt.plotting import FileDialog
            FileDialog(
                self.plotter.app_window,
                callback=partial(_save_ndarray_img, img=img),
            )

        self._tool_bar_add_button(
            name=name,
            desc=desc,
            func=_screenshot,
        )

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
        self.menu_bar = QMenuBar()
        self.menu_bar.setNativeMenuBar(False)
        window = self._window if window is None else window
        window.setMenuBar(self.menu_bar)

    def _menu_add_submenu(self, name, desc):
        self._menus[name] = self.menu_bar.addMenu(desc)

    def _menu_add_button(self, menu_name, name, desc, func):
        menu = self._menus[menu_name]
        self._menu_actions[name] = menu.addAction(desc, func)


class _QtStatusBar(_AbstractStatusBar):
    def _status_bar_initialize(self, window=None):
        window = self._window if window is None else window
        self.status_bar = window.statusBar()

    def _status_bar_add_label(self, value, stretch=0):
        widget = QLabel(value)
        self.status_bar.layout().addWidget(widget, stretch)
        return widget

    def _status_bar_add_progress_bar(self, stretch=0):
        widget = QProgressBar()
        self.status_bar.layout().addWidget(widget, stretch)
        return widget


class _QtPlayback(_AbstractPlayback):
    def _playback_initialize(self, func, timeout):
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
        dock, dock_layout = _create_dock_widget(
            self._window, "Traces", Qt.BottomDockWidgetArea)
        dock_layout.addWidget(canvas)

    def _window_get_cursor(self):
        return self._interactor.cursor()

    def _window_set_cursor(self, cursor):
        self._interactor.setCursor(cursor)

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
        if adjust_mpl:
            mpl_h = int(round((sz[1] * self._interactor_fraction) /
                              (1 - self._interactor_fraction)))
            self._mplcanvas.canvas.setMinimumSize(sz[0], mpl_h)
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
            self._process_events()
            self._process_events()
            # 4. Resize the window and interactor to the correct size
            #    (not sure why, but this is required on macOS at least)
            self._interactor.window_size = (win_sz.width(), win_sz.height())
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


class _QtWidget(_AbstractWidget):
    def set_value(self, value):
        if hasattr(self._widget, "setValue"):
            self._widget.setValue(value)
        elif hasattr(self._widget, "setCurrentText"):
            self._widget.setCurrentText(value)
        else:
            assert hasattr(self._widget, "setText")
            self._widget.setText(value)

    def get_value(self):
        if hasattr(self._widget, "value"):
            return self._widget.value()
        elif hasattr(self._widget, "currentText"):
            return self._widget.currentText()
        elif hasattr(self._widget, "text"):
            return self._widget.text()


class _Renderer(_PyVistaRenderer, _QtDock, _QtToolBar, _QtMenuBar,
                _QtStatusBar, _QtWindow, _QtPlayback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_initialize()

    def show(self):
        super().show()
        with _qt_disable_paint(self.plotter):
            with self._window_ensure_minimum_sizes():
                self.plotter.app_window.show()
        self.plotter.update()


def _create_dock_widget(window, name, area):
    dock = QDockWidget()
    scroll = QScrollArea(dock)
    dock.setWidget(scroll)
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
