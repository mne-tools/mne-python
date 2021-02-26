"""ToolBar implemented with Qt."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from .abstract_tool_bar import _AbstractToolBar
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QSizePolicy, QWidget


class _QtToolBar(_AbstractToolBar):
    def _tool_bar_load_icons(self):
        from ._utils import _init_qt_resources
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

    def _tool_bar_initialize(self, name="default"):
        self.actions = dict()
        self.tool_bar = self.plotter.app_window.addToolBar(name)

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
