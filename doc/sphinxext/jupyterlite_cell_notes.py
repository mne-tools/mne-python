"""Per-notebook fixups applied to the JupyterLite copies only.

:func:`note_unrunnable_cells` does two things to every notebook sphinx-gallery
copies into the JupyterLite contents:

1. Prepends the setup cell that installs MNE and patches the browser
   environment (see :mod:`jupyterlite_setup_cell`).
2. Swaps any cell that cannot run in the browser for a note that keeps the code
   in view and says what it needs instead; a cell here and there is blocked
   even though the rest of its notebook runs, and dropping a whole page from
   the launcher over one cell costs more than it saves.

Both are deliberately done here rather than through ``first_notebook_cell``,
which sphinx-gallery applies while *generating* the notebook and therefore also
writes into the ``.ipynb`` offered for download, where ``piplite`` does not
exist and the notebook would fail on its first cell. Doing it at copy time
keeps the download and the rendered page exactly as the docs built them.

It lives here rather than in ``conf.py`` because ``sphinx_gallery_conf`` has to
stay JSON-serializable (``sphinx.config.is_serializable`` rejects functions), so
the config names the dotted path and sphinx-gallery imports it.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import sphinx.util.logging
from jupyterlite_setup_cell import LITE_SETUP_CELL

# not mne_doc_utils.sphinx_logger: that module pulls in mne and pyvista, which
# this one has no use for
logger = sphinx.util.logging.getLogger("mne")

# first line of LITE_SETUP_CELL, used to spot a cell that is already there
_SETUP_MARKER = LITE_SETUP_CELL.strip().split("\n", 1)[0]

# (notebook path suffix, substring identifying the cell, replacement markdown).
# Keep this short: a page that is mostly unavailable belongs in
# JUPYTERLITE_EXCLUDE instead of here.
CELL_NOTES = (
    (
        "forward/20_source_alignment.ipynb",
        "mne.gui.coregistration",
        "**This cell does not run in the browser.**\n"
        "\n"
        "`mne.gui.coregistration` sets the fiducials by clicking on the scalp\n"
        "surface, and the vtk.js renderer used here draws scenes without a\n"
        "picker, so there is nothing for those clicks to hit. Run it from a\n"
        "local MNE install instead:\n"
        "\n"
        "```python\n"
        'mne.gui.coregistration(subject="sample", subjects_dir=subjects_dir)\n'
        "```\n"
        "\n"
        "The video above walks through the same steps, and the rest of this\n"
        "notebook runs normally.\n",
    ),
)


def note_unrunnable_cells(notebook_content, notebook_filename):
    """Add the setup cell and note the cells that cannot run in the browser.

    Parameters
    ----------
    notebook_content : dict
        The parsed notebook, modified in place.
    notebook_filename : path-like
        Where the notebook will be written inside the JupyterLite contents.
    """
    # setdefault, not get: a missing key would otherwise hand back a throwaway
    # list and the insert would silently not stick
    cells = notebook_content.setdefault("cells", [])
    # stale .ipynb from an earlier build can already carry the cell; adding a
    # second one would install everything twice
    already = cells and _SETUP_MARKER in "".join(cells[0].get("source", []))
    if not already:
        cells.insert(
            0,
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"collapsed": False},
                "outputs": [],
                # .strip() to match what sphinx-gallery's add_code_cell wrote
                # while this went through first_notebook_cell
                "source": [LITE_SETUP_CELL.strip()],
            },
        )
    path = str(notebook_filename).replace(os.sep, "/")
    for suffix, needle, note in CELL_NOTES:
        if not path.endswith(suffix):
            continue
        for cell in cells:
            # prose mentions the same function, so only rewrite real code
            if cell.get("cell_type") != "code":
                continue
            if needle not in "".join(cell.get("source", [])):
                continue
            cell["cell_type"] = "markdown"
            cell["source"] = [note]
            cell["metadata"] = {}
            # markdown cells carry neither of these, and nbformat rejects them
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
            logger.info(f"[JupyterLite]   {suffix}: noted {needle} cell")
