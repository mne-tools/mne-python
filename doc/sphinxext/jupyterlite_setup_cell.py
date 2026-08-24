"""The setup cell prepended to every JupyterLite notebook.

It installs MNE into the browser kernel and patches what Pyodide does not
provide: data fetching over HTTP, the readers that expect files already on
disk, and the 3D renderer. The cell itself lives in ``_lite_setup_cell.py`` as
ordinary Python, so ruff lints and formats it; this module only reads that file
and exposes it as the string the browser kernel needs.

The docs build prepends it only to the notebooks copied into the JupyterLite
contents. It deliberately does NOT go through ``first_notebook_cell``: that is
applied when the notebook is generated, so it would also land in the ``.ipynb``
offered for download, where ``piplite`` does not exist and the notebook would
fail on its first cell.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from jupyterlite_lite_renderer import LITE_RENDERER_CELL

_SOURCE = Path(__file__).parent / "_lite_setup_cell.py"
# Everything after the banner is what the notebook runs. The license header and
# the ruff directives above it belong to the file, not to the cell.
_BANNER = "# --- JupyterLite setup cell"

_text = _SOURCE.read_text()
if _BANNER not in _text:
    raise RuntimeError(f"{_SOURCE.name} is missing the {_BANNER!r} banner")
_body = _text[_text.index(_BANNER) :]
_body = _body[_body.index("\n") + 1 :]

# The renderer goes last, so MNE is already imported by the time it runs; see
# jupyterlite_lite_renderer.py.
LITE_SETUP_CELL = _body + LITE_RENDERER_CELL
