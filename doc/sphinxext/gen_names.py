# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from os import path as op


def setup(app):
    app.connect("builder-inited", generate_name_links_rst)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def generate_name_links_rst(app=None):
    if "linkcheck" not in str(app.builder).lower():
        return
    out_dir = op.abspath(op.join(op.dirname(__file__), "..", "generated"))
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    out_fname = op.join(out_dir, "_names.rst")
    names_path = op.abspath(
        op.join(os.path.dirname(__file__), "..", "changes", "names.inc")
    )
    with open(out_fname, "w", encoding="utf8") as fout:
        fout.write(":orphan:\n\n")
        with open(names_path) as fin:
            for line in fin:
                if line.startswith(".. _"):
                    fout.write(f"- {line[4:]}")
