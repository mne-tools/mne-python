"""The ``notebook`` 3D backend: VTK drawn through PyVista and trame."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import asyncio
import threading
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
        if jupyter_backend == "none":  # asked to display nothing (the app path)
            return super().show(*args, jupyter_backend="none", **kwargs)
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
        try:  # the kernel's loop, on the main thread, which owns the GL context
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
        self._render_pending = False
        try:
            return self._app_widget()
        except (ImportError, RuntimeError):  # trame-pyvista without the app
            return self.plotter.show(return_viewer=True)

    def _app_widget(self):
        """Return the widget of trame-pyvista's application showing the plotter.

        Server rendering streamed by trame-rca, with the application's own
        toolbar and a switch to VTK compiled to WebAssembly in the browser.
        """
        import pyvista
        from trame_pyvista import module
        from trame_pyvista.apps import SimpleViewer
        from trame_pyvista.jupyter import elegantly_launch
        from trame_pyvista.widgets import get_server

        name = pyvista.global_theme.trame.jupyter_server_name
        server = get_server(name)
        # the application's styles are served only if enabled before launching
        server.enable_module(module)
        if not server.running:
            elegantly_launch(name)
        # the bookkeeping show() does (first render, no longer "first time"),
        # without PyVista's Jupyter backend displaying anything
        self.plotter.show(jupyter_backend="none", return_viewer=False, auto_close=False)
        self._trame_app = SimpleViewer(self.plotter, name, mode="remote")
        width, height = self.plotter.window_size  # the iframe's own size, in px
        self._trame_app.ui.iframe_style = (
            f"width: {width}px; height: {height}px; border: none"
        )
        return self._trame_app.ui.ipywidget

    def _update(self):
        # JupyterLab (>= 4.4) delivers widget events to kernel subshells, which
        # run on their own threads; VTK can only render on the thread that owns
        # its OpenGL context, so render from the main loop instead. Brain asks
        # several times per event, so once.
        loop = getattr(self, "_main_loop", None)
        if loop is None or threading.current_thread() is threading.main_thread():
            super()._update()
        elif not self._render_pending:
            self._render_pending = True
            loop.call_soon_threadsafe(self._update_from_main)

    def _update_from_main(self):
        self._render_pending = False
        super()._update()


_testing_context = nullcontext
