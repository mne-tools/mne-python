from abc import ABC, abstractmethod


class _AbstractDock(ABC):
    @abstractmethod
    def _dock_initialize(self):
        pass

    @abstractmethod
    def _dock_finalize(self):
        pass

    @abstractmethod
    def _dock_show(self):
        pass

    @abstractmethod
    def _dock_hide(self):
        pass

    @abstractmethod
    def _dock_add_stretch(self, layout):
        pass

    @abstractmethod
    def _dock_add_layout(self, vertical=True):
        pass

    @abstractmethod
    def _dock_add_label(self, value, align=False, layout=None):
        pass

    @abstractmethod
    def _dock_add_button(self, name, callback, layout=None):
        pass

    @abstractmethod
    def _dock_add_text(self, widget_name, value, callback, validator=None,
                       layout=None):
        pass

    @abstractmethod
    def _dock_add_slider(self, label_name, value, rng, callback,
                         compact=True, double=False, layout=None):
        pass

    @abstractmethod
    def _dock_add_spin_box(self, label_name, value, rng, callback,
                           compact=True, double=True, layout=None):
        pass

    @abstractmethod
    def _dock_add_combo_box(self, label_name, value, rng,
                            callback, compact=True, layout=None):
        pass

    @abstractmethod
    def _dock_add_group_box(self, name, layout=None):
        pass
