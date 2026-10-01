"""An off-screen PyVista backend for MNE's 3D renderer.

Draws with a plain :class:`pyvista.Plotter` in off-screen mode: no Qt, no
notebook widgets and no window. Figures are consumed through
:meth:`~mne.viz.Brain.screenshot` and :meth:`~mne.viz.Brain.save_movie`, which
is what reports, documentation builds and headless servers need. Selected with
``mne.viz.set_3d_backend("offscreen")``.

Everything that draws works, including :class:`mne.viz.Brain`; only the
interactive widgets do not, so :meth:`mne.viz.Brain.setup_time_viewer` raises.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextlib import nullcontext
from functools import cached_property

import pyvista

from ._pyvista import (
    _check_3d_figure,  # noqa: F401
    _clear_3d_figure,  # noqa: F401
    _close_3d_figure,  # noqa: F401
    _close_all,  # noqa: F401
    _PyVistaRenderer,
    _set_3d_title,  # noqa: F401
    _set_3d_view,  # noqa: F401
    _take_3d_screenshot,  # noqa: F401
)
from ._utils import _CloseCallbackMixin, _run_close_callbacks


class _Plotter(pyvista.Plotter):
    """A plain Plotter that runs close callbacks, as a Qt window does on closing.

    The plotter runs them rather than the renderer because ``close_3d_figure``
    and ``close_all_3d_figures`` close the plotter without going through one.
    """

    @cached_property
    def _close_callbacks(self):
        return {False: [], True: []}  # keyed by ``after``

    def close(self):
        first = not self._closed
        if first:
            _run_close_callbacks(self._close_callbacks, after=False)
        super().close()
        if first:
            _run_close_callbacks(self._close_callbacks, after=True)


class _Renderer(_CloseCallbackMixin, _PyVistaRenderer):
    _kind = "offscreen"
    _plotter_class = _Plotter

    @property
    def _close_callbacks(self):
        return self.plotter._close_callbacks

    def show(self):
        # pyvista.Plotter.show() renders once and then closes the plotter,
        # which would leave nothing to screenshot; just render
        self._update()

    def _window_set_theme(self, theme):
        pass  # there is no window


_testing_context = nullcontext  # the plotter is always off screen
