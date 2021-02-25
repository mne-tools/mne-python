from abc import ABC, abstractmethod


class _AbstractDock(ABC):
    @abstractmethod
    def _initialize_dock(self):
        pass

    @abstractmethod
    def _finalize_dock(self):
        pass

    @abstractmethod
    def _add_dock_stretch(self, layout):
        pass

    @abstractmethod
    def _add_dock_layout(self, vertical=True):
        pass

    @abstractmethod
    def _add_dock_label(self, value, align=False, layout=None):
        pass

    @abstractmethod
    def _add_dock_button(self, name, callback, layout=None):
        pass

    @abstractmethod
    def _add_dock_text(self, widget_name, value, callback, validator=None,
                       layout=None):
        pass

    @abstractmethod
    def _add_dock_slider(self, label_name, value, rng, callback,
                         compact=True, double=False, layout=None):
        pass

    @abstractmethod
    def _add_dock_spin_box(self, label_name, value, rng, callback,
                           compact=True, double=True, layout=None):
        pass

    @abstractmethod
    def _add_dock_combo_box(self, label_name, value, rng,
                            callback, compact=True, layout=None):
        pass

    @abstractmethod
    def _add_dock_group_box(self, name, layout=None):
        pass

    @abstractmethod
    def _show_dock(self):
        pass

    @abstractmethod
    def _hide_dock(self):
        pass
