"""A pyvista-js drawing backend for MNE's 3D renderer, for JupyterLite.

MNE's 3D functions (``plot_alignment``, ``plot_bem``, ``plot_sparse_source_estimates``,
``SourceSpaces.plot``, ...) all build their figure the same way: they do their own
geometry and coordinate-frame work in numpy, then hand the result to a renderer
obtained from ``mne.viz.backends.renderer._get_renderer``. Only that last step needs
VTK, and VTK cannot load in WebAssembly.

So instead of reimplementing those functions one by one, this module supplies a
renderer that draws with pyvista-js (vtk.js) and patches the factory, along with the
``renderer.backend`` global that ``set_3d_view`` and the other scene-level helpers
read directly. MNE then does all of the transform math itself, which matters
because getting a head/MRI/device transform subtly wrong produces a
plausible-looking picture with the sensors in the wrong place, and several of these
tutorials are specifically *about* coordinate alignment.

What is supported: meshes, surfaces, spheres, tubes and glyphs, enough for the
static figures the docs render. What is not: the interactive ``Brain`` time viewer,
which additionally needs dock widgets and toolbars, and scalar colormaps, which
pyvista-js 0.15 does not have (scalars fall back to a solid color).

The renderer itself lives in ``_lite_renderer_cell.py`` as ordinary Python, so ruff
lints and formats it like any other module. This module only reads that file and
exposes it as a string, which is the form the browser kernel needs: it is appended
to ``LITE_SETUP_CELL`` in ``jupyterlite_setup_cell.py``, which the docs build
prepends to each JupyterLite notebook.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

_SOURCE = Path(__file__).parent / "_lite_renderer_cell.py"
# Everything from the banner onwards is what the notebook runs. The license header
# above it belongs to the file rather than to the cell, so it is left behind.
_BANNER = "# --- pyvista-js drawing backend"

_text = _SOURCE.read_text()
if _BANNER not in _text:
    raise RuntimeError(f"{_SOURCE.name} is missing the {_BANNER!r} banner")
LITE_RENDERER_CELL = "\n" + _text[_text.index(_BANNER) :]
