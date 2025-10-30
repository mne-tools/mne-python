# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re

from mne_doc_utils import sphinx_logger


def setup(app):
    app.connect("source-read", check_directive_formatting)
    app.connect("autodoc-process-docstring", check_directive_formatting)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def check_directive_formatting(*args):
    """Check that directives are not missing a space.

    For args, see Sphinx events 'source-read' and 'autodoc-process-docstring'.
    """
    if len(args) == 3:  # from source-read
        source_type = "File"
        name = args[1]
        source = args[2][0].split("\n")
    else:  # from autodoc-process-docstring
        source_type = "Docstring"
        name = args[2]
        source = args[5]

    for idx, line in enumerate(source):
        # check for missing space after '..'
        missing = re.search(r"\.\.[a-zA-Z]+::", line)
        if missing is not None:
            sphinx_logger.warning(
                f"{source_type} '{name}' is missing a space after '..' in the "
                f"directive '{missing.group()}'"
            )

        # check for missing preceding blank line
        if idx == 0:
            continue
        dir_pattern = r"\.\. [a-zA-Z]+::"
        head_pattern = r"^[-|=|\^]+$"
        directive = re.search(dir_pattern, line)
        if directive is not None:
            line_prev = source[idx - 1].strip()
            if (
                line_prev != ""  # not an empty line
                and not re.search(head_pattern, line_prev)  # not after a header
                and not re.search(dir_pattern, line_prev)  # not after directive
            ):
                # check if previous line is part of another directive
                bad = True
                for line_prev in reversed(source[: idx - 1]):
                    line_prev = line_prev.strip()
                    if line_prev == "" or re.search(head_pattern, line_prev):
                        # if a blank line or header, is not part of another directive
                        break
                    if re.search(dir_pattern, line_prev):
                        bad = False  # is part of another directive
                        break
                    # or keep going until we reach the first line (so must be bad)
                if bad:
                    sphinx_logger.warning(
                        f"{source_type} '{name}' is missing a blank line before the "
                        f"directive '{directive.group()}'"
                    )
