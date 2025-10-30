# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re

from mne_doc_utils import sphinx_logger


def setup(app):
    app.connect("source-read", check_directive_formatting)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def check_directive_formatting(app=None, docname=None, content=None):
    """Check that Sphinx directives are properly formatted."""
    directives = [
        "attention",
        "caution",
        "danger",
        "error",
        "hint",
        "important",
        "note",
        "tip",
        "warning",
        "admonition",
        "seealso",
        "versionadded",
        "version-added",
        "versionchanged",
        "version-changed",
        "deprecated",
        "version-deprecated",
        "versionremoved",
        "version-removed",
    ]

    missing_space = re.search(r"\.\.[a-zA-Z]+::", content[0])
    if missing_space is not None:
        sphinx_logger.warning(
            f"File '{docname}' is missing a space after '..' in the directive "
            f"'{missing_space.group()}'"
        )

    content_stripped = content[0].replace(" ", "")
    for match in re.finditer(rf"\.\.({('|').join(directives)})::", content_stripped):
        prematch = content_stripped[match.start() - 2 : match.start()]
        if (
            prematch != "\n\n"  # no preceding blank line
            and not re.search(r"(-|=|\^)\n", prematch)  # and not after a section header
        ):
            sphinx_logger.warning(
                f"File '{docname}' is missing a blank line before the directive "
                f"'{missing_space.group()}'"
            )
