"""The setup cell prepended to every JupyterLite notebook.

It hands over to ``mne.viz.backends._jupyterlite.setup_notebook``, which
patches what Pyodide does not provide. ``sys.platform`` is ``"emscripten"``
only inside Pyodide, so the cell is a no-op in a local kernel and a notebook
downloaded from inside JupyterLite runs unchanged there.

The docs build prepends it only to the notebooks copied into the JupyterLite
contents, not through ``first_notebook_cell``, which would also put it in the
``.ipynb`` offered for download.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ast

from jupyterlite_lock_specs import mne_pypi_spec

_INSTALL_MNE = """\
    import piplite

    await piplite.install(["mne"], keep_going=True)
"""

# (which ends with one) is substituted in
LITE_SETUP_CELL = f"""\
# 💡 Added by the docs build: adapts MNE to Pyodide. Does nothing outside
# JupyterLite.
import sys

if sys.platform == "emscripten":
{_INSTALL_MNE if mne_pypi_spec() is None else ""}\
    from mne.viz.backends._jupyterlite import setup_notebook

    setup_notebook()
"""
# nothing else runs this before a reader does, so at least make sure it parses
compile(
    LITE_SETUP_CELL, "lite_setup_cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
)
