"""Basic tool bar support."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from abc import ABC, abstractmethod


class _AbstractToolBar(ABC):
    @abstractmethod
    def _tool_bar_load_icons(self):
        pass

    @abstractmethod
    def _tool_bar_initialize(self, name="default"):
        pass

    @abstractmethod
    def _tool_bar_finalize(self):
        pass

    @abstractmethod
    def _tool_bar_add_button(self, name, desc, func, icon_name=None):
        pass

    @abstractmethod
    def _tool_bar_update_button_icon(self, name, icon_name):
        pass

    @abstractmethod
    def _tool_bar_add_text(self, name, value, placeholder):
        pass

    @abstractmethod
    def _tool_bar_add_spacer(self):
        pass
