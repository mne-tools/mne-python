"""The setup cell prepended to every JupyterLite notebook.

It installs MNE into the browser kernel and hands over to
``mne.viz.backends._jupyterlite.setup_notebook``, which patches what Pyodide
does not provide. piplite (not micropip) prefers the development MNE wheel
bundled with the docs over PyPI, and ``keep_going`` reports a dependency with
no wheel instead of aborting. ``sys.platform`` is ``"emscripten"`` only inside
Pyodide, so the cell is a no-op in a local kernel and a notebook downloaded
from inside JupyterLite runs unchanged there.

The docs build prepends it only to the notebooks copied into the JupyterLite
contents, not through ``first_notebook_cell``, which would also put it in the
``.ipynb`` offered for download.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# ruff: noqa: E501  # the install list stays on one line so the cell reads short

import ast

LITE_SETUP_CELL = """\
# 💡 Added by the docs build: installs MNE into the browser kernel and adapts
# it to Pyodide. Does nothing outside JupyterLite.
import sys

if sys.platform == "emscripten":
    import piplite

    await piplite.install(["mne", "scikit-learn", "joblib", "pandas", "seaborn", "mne-connectivity", "nibabel", "pyvista-js", "pyxdf", "mffpy", "python-picard"], keep_going=True)  # noqa: E501
    from mne.viz.backends._jupyterlite import setup_notebook

    setup_notebook()
"""
# nothing else runs this before a reader does, so at least make sure it parses
compile(
    LITE_SETUP_CELL, "lite_setup_cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
)
