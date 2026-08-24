"""Turn on MNE's pyvista-js 3D renderer inside the JupyterLite kernel.

VTK has no WebAssembly build, so the browser draws with pyvista-js instead. The
renderer itself is ordinary library code in ``mne/viz/backends/_lite.py``; this
module only exposes the few lines of notebook code that switch MNE over to it.
``LITE_RENDERER_CELL`` is appended to ``LITE_SETUP_CELL`` in
``jupyterlite_setup_cell.py``, which the docs build prepends to each notebook.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

LITE_RENDERER_CELL = """
# Using pyvista-js (vtk.js) to draw MNE's 3D rendering in JupyterLite.
# See mne/viz/backends/_lite.py for more details.
try:
    from mne.viz.backends._lite import _activate as _mne_activate_lite_renderer

    _mne_activate_lite_renderer()
except Exception as _e:
    print("[JupyterLite] could not install the pyvista-js renderer: " + repr(_e))
"""
