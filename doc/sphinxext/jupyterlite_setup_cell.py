"""The setup cell prepended to every JupyterLite notebook.

It installs MNE into the browser kernel and patches what Pyodide does not
provide: data fetching over HTTP, the readers that expect files already on
disk, and the 3D renderer. The cell lives in ``_lite_setup_cell.py`` and
``_lite_setup_cell_3d.py`` as ordinary Python, so ruff lints and formats it;
this module only reads those files and joins them into the string the browser
kernel needs. The 3D half is kept separate because it stands in for MNE's
Brain/VTK stack and is the part most likely to change as pyvista-js gains
features upstream.

The docs build prepends it only to the notebooks copied into the JupyterLite
contents. It deliberately does NOT go through ``first_notebook_cell``: that is
applied when the notebook is generated, so it would also land in the ``.ipynb``
offered for download, where ``piplite`` does not exist and the notebook would
fail on its first cell.

The other direction is covered in the cell itself: a notebook downloaded from
inside JupyterLite does carry the cell, and it says to delete it before running
locally, for the same reason.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from jupyterlite_lite_renderer import LITE_RENDERER_CELL

# Each source file read below is split at this banner: everything after it is
# what the notebook runs, and what sits above it in that file (license header,
# ruff directives, notes for whoever edits it) stays behind.
_BANNER = "# --- JupyterLite setup cell"


def _read(name):
    _source = Path(__file__).parent / name
    _text = _source.read_text()
    if _BANNER not in _text:
        raise RuntimeError(f"{_source.name} is missing the {_BANNER!r} banner")
    _body = _text[_text.index(_BANNER) :]
    return _body[_body.index("\n") + 1 :]


# Order matters: the 3D half reads the matplotlib shim the base half installs,
# and the renderer goes last so MNE is already imported by the time it runs.
LITE_SETUP_CELL = (
    _read("_lite_setup_cell.py") + _read("_lite_setup_cell_3d.py") + LITE_RENDERER_CELL
)
