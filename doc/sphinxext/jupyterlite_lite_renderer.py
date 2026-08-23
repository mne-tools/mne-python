"""Turn on MNE's pyvista-js 3D renderer inside the JupyterLite kernel.

VTK cannot load in WebAssembly, so MNE's 3D output would otherwise be
unavailable in the browser. The renderer that replaces it lives in MNE itself,
at ``mne/viz/backends/_lite.py``, so it is ordinary library code: linted,
formatted and unit tested like any other module, and shipped in the wheel the
browser kernel installs.

That leaves this module with one job. It exposes the few lines of notebook code
that import the renderer and switch MNE over to it. ``LITE_RENDERER_CELL`` is
appended to ``LITE_SETUP_CELL`` in ``jupyterlite_setup_cell.py``, which the docs
build prepends to each JupyterLite notebook.

The cell degrades quietly: if the installed MNE predates the renderer, or
pyvista-js is missing, the notebook prints why and carries on with everything
that does not need 3D.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

LITE_RENDERER_CELL = """
# Draw MNE's 3D figures with pyvista-js (vtk.js). VTK has no WebAssembly build,
# so this swaps only the drawing step; MNE still does its own geometry and
# coordinate-frame work. See mne/viz/backends/_lite.py.
try:
    from mne.viz.backends._lite import _activate as _mne_activate_lite_renderer

    _mne_activate_lite_renderer()
except Exception as _e:
    print("[JupyterLite] could not install the pyvista-js renderer: " + repr(_e))
"""
