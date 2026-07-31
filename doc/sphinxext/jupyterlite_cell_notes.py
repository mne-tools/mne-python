"""Notes for notebook cells that cannot run in the JupyterLite kernel.

A cell here and there cannot run in the browser even though the rest of its
notebook can, and dropping a whole page from the launcher over one cell costs
more than it saves. :func:`note_unrunnable_cells` swaps just that cell for a
note that keeps the code in view and says what it needs instead.

sphinx-gallery calls this for each notebook it copies into the JupyterLite
contents, so only the browser copy changes -- the notebook offered for download
stays exactly as the docs built it.

It lives here rather than in ``conf.py`` because ``sphinx_gallery_conf`` has to
stay JSON-serializable (``sphinx.config.is_serializable`` rejects functions), so
the config names the dotted path and sphinx-gallery imports it.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import sphinx.util.logging

# not mne_doc_utils.sphinx_logger: that module pulls in mne and pyvista, which
# this one has no use for
logger = sphinx.util.logging.getLogger("mne")

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
    """Turn cells that cannot run in the browser into an explanatory note.

    Parameters
    ----------
    notebook_content : dict
        The parsed notebook, modified in place.
    notebook_filename : path-like
        Where the notebook will be written inside the JupyterLite contents.
    """
    path = str(notebook_filename).replace(os.sep, "/")
    for suffix, needle, note in CELL_NOTES:
        if not path.endswith(suffix):
            continue
        for cell in notebook_content.get("cells", []):
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
