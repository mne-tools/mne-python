"""Qt implementation of _Renderer and GUI."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from contextlib import contextmanager

import pyvista

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDoubleSpinBox, QGroupBox,
                             QHBoxLayout, QLabel, QToolButton, QMenuBar,
                             QSlider, QSpinBox, QVBoxLayout, QWidget,
                             QSizePolicy, QScrollArea, QStyle,
                             QStyleOptionSlider)

from ._pyvista import _PyVistaRenderer
from ._pyvista import (_close_all, _close_3d_figure, _check_3d_figure,  # noqa: F401,E501 analysis:ignore
                       _set_3d_view, _set_3d_title, _take_3d_screenshot)  # noqa: F401,E501 analysis:ignore
from ._abstract import _AbstractDock, _AbstractToolBar
from ._utils import _init_qt_resources


class _QtDock(_AbstractDock):
    def _dock_initialize(self, window):
        self.dock = QDockWidget()
        self.scroll = QScrollArea(self.dock)
        self.dock.setWidget(self.scroll)
        widget = QWidget(self.scroll)
        self.scroll.setWidget(widget)
        self.scroll.setWidgetResizable(True)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        window.addDockWidget(Qt.LeftDockWidgetArea, self.dock)
        self.dock_layout = QVBoxLayout()
        widget.setLayout(self.dock_layout)

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
        layout.addWidget(widget)
        return widget

    def _dock_add_button(self, name, callback, layout=None):
        layout = self.dock_layout if layout is None else layout
        # If we want one with text instead of an icon, we should use
        # QPushButton(name)
        widget = QToolButton()
        widget.clicked.connect(callback)
        widget.setText(name)
        layout.addWidget(widget)
        return widget

    def _dock_named_layout(self, name, layout, compact):
        layout = self.dock_layout if layout is None else layout
        if name is not None:
            hlayout = self._dock_add_layout(not compact)
            self._dock_add_label(
                value=name, align=not compact, layout=hlayout)
            layout.addLayout(hlayout)
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
        layout.addWidget(widget)
        return widget

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
        layout.addWidget(widget)
        return widget

    def _dock_add_combo_box(self, name, value, rng,
                            callback, compact=True, layout=None):
        layout = self._dock_named_layout(name, layout, compact)
        widget = QComboBox()
        widget.addItems(rng)
        widget.setCurrentText(value)
        widget.currentTextChanged.connect(callback)
        widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        layout.addWidget(widget)
        return widget

    def _dock_add_group_box(self, name, layout=None):
        layout = self.dock_layout if layout is None else layout
        hlayout = QVBoxLayout()
        widget = QGroupBox(name)
        widget.setLayout(hlayout)
        layout.addWidget(widget)
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


class _QtToolBar(_AbstractToolBar):
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

    def _tool_bar_initialize(self, window, name="default"):
        self.actions = dict()
        self.tool_bar = window.addToolBar(name)

    def _tool_bar_finalize(self):
        pass

    def _tool_bar_add_button(self, name, desc, func, icon_name=None):
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        self.actions[name] = self.tool_bar.addAction(icon, desc, func)

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].setIcon(self.icons[icon_name])

    def _tool_bar_add_text(self, name, value, placeholder):
        pass

    def _tool_bar_add_spacer(self):
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tool_bar.addWidget(spacer)


class _QtMenuBar():
    def _menu_initialize(self, window):
        self._menus = dict()
        self._menu_actions = dict()
        self.menu_bar = QMenuBar()
        self.menu_bar.setNativeMenuBar(False)
        window.setMenuBar(self.menu_bar)

    def _menu_add_submenu(self, name, desc):
        self._menus[name] = self.menu_bar.addMenu(desc)

    def _menu_add_button(self, menu_name, name, desc, func):
        menu = self._menus[menu_name]
        self._menu_actions[name] = menu.addAction(desc, func)


class _Renderer(_PyVistaRenderer, _QtDock, _QtToolBar, _QtMenuBar):
    pass


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
