# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

from ...fixes import nullcontext
from ._pyvista import _Renderer as _PyVistaRenderer
from ._pyvista import \
    _close_all, _set_3d_view, _set_3d_title  # noqa: F401 analysis:ignore


class _Renderer(_PyVistaRenderer):
    def __init__(self, *args, **kwargs):
        self.disp = None
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)

    def show(self):
        from IPython.display import display
        self.figure.display = self.plotter.show(use_ipyvtk=True,
                                                return_viewer=True)
        display(self.figure.display)
        return self.scene()


_testing_context = nullcontext
