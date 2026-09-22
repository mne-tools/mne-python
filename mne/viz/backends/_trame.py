"""The ``notebook`` 3D backend: VTK drawn through PyVista and trame."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import warnings
from contextlib import contextmanager, nullcontext

from IPython.display import display
from ipywidgets import Widget

from ._notebook import _IpyRenderer
from ._pyvista import (
    Plotter,
    _check_3d_figure,  # noqa: F401
    _clear_3d_figure,  # noqa: F401
    _close_3d_figure,  # noqa: F401
    _close_all,  # noqa: F401
    _PyVistaRenderer,
    _set_3d_title,  # noqa: F401
    _set_3d_view,  # noqa: F401
    _take_3d_screenshot,  # noqa: F401
)
from ._utils import _notebook_vtk_works

_JUPYTER_BACKEND = "trame"


class _NotebookPlotter(Plotter):
    """PyVista ``Plotter`` for the notebook backend.

    Validate the object returned by show, as PyVista silently falls back static PIL
    when the trame Jupyter backend cannot be loaded.
    """

    def show(
        self, *args, jupyter_backend=_JUPYTER_BACKEND, return_viewer=False, **kwargs
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            viewer = super().show(
                *args,
                jupyter_backend=jupyter_backend,
                return_viewer=return_viewer,
                **kwargs,
            )
        if not isinstance(viewer, Widget):
            reasons = "\n".join(
                f"- {w.message}"
                for w in caught
                if any(key in str(w.message) for key in ("backend", "trame", "static"))
            )
            raise RuntimeError(
                f'The notebook 3D backend is not functional: the "{jupyter_backend}" '
                "PyVista Jupyter backend returned a "
                f"{type(viewer).__module__}.{type(viewer).__qualname__} instead of an "
                "interactive widget. This usually means the installed trame packages "
                "(trame, trame-vtk, trame-vuetify, trame-pyvista) are missing or "
                "mutually incompatible."
                + (f"\n\nPyVista reported:\n{reasons}" if reasons else "")
            )
        if return_viewer:
            return viewer


class _3DRenderer(_PyVistaRenderer):
    _kind = "notebook"

    def __init__(self, *args, **kwargs):
        kwargs["notebook"] = True
        super().__init__(*args, **kwargs)
        if "show" in kwargs and kwargs["show"]:
            self.show()

    @contextmanager
    def _ensure_minimum_sizes(self):
        yield

    def show(self):
        viewer = self.plotter.show(return_viewer=True)
        viewer.layout.width = None  # unlock the fixed layout
        display(viewer)


class _Renderer(_IpyRenderer, _PyVistaRenderer):
    """VTK drawing through PyVista and trame, with the shared ipywidgets GUI."""

    _kind = "notebook"

    def __init__(self, *args, **kwargs):
        kwargs["notebook"] = True
        fullscreen = kwargs.pop("fullscreen", False)
        if not _notebook_vtk_works():
            raise RuntimeError(
                "Using the notebook backend on Linux requires a compatible "
                "VTK setup. Consider using Xfvb or xvfb-run to set up a "
                "working virtual display, or install VTK with OSMesa enabled."
            )
        super().__init__(*args, **kwargs)
        self._window_initialize(fullscreen=fullscreen)

    def _viewer_widget(self):
        return self.plotter.show(return_viewer=True)


_testing_context = nullcontext
