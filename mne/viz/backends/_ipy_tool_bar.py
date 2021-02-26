"""ToolBar implemented with ipywidgets."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from .abstract_tool_bar import _AbstractToolBar
from ._utils import _ipy_add_widget
from ipywidgets import HBox, Button, Text


class _IpyToolBar(_AbstractToolBar):
    def _tool_bar_load_icons(self):
        self.icons = dict()
        self.icons["help"] = None
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

    def _tool_bar_initialize(self, name="default"):
        self.actions = dict()
        self.tool_bar = HBox()

    def _tool_bar_finalize(self):
        pass

    def _tool_bar_add_button(self, name, desc, func, icon_name=None):
        icon_name = name if icon_name is None else icon_name
        icon = self.icons[icon_name]
        if icon is None:
            return
        widget = Button(tooltip=desc, icon=icon)
        widget.on_click(lambda x: func())
        _ipy_add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_update_button_icon(self, name, icon_name):
        self.actions[name].icon = self.icons[icon_name]

    def _tool_bar_add_text(self, name, value, placeholder):
        widget = Text(value=value, placeholder=placeholder)
        _ipy_add_widget(self.tool_bar, widget)
        self.actions[name] = widget

    def _tool_bar_add_spacer(self):
        pass
