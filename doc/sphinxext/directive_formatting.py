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
    # Extract relevant info from args
    if len(args) == 3:  # from source-read
        source_type = "File"
        name = args[1]
        source = args[2][0]
        source_concat = source  # content already a single string
    elif len(args) == 6:  # from autodoc-process-docstring
        source_type = "Docstring"
        name = args[2]
        source = args[5]
        source_concat = "\n".join(source)  # combine lines into single string
    else:
        raise RuntimeError("Unexpected number of arguments from Sphinx event")

    # Check if any directives are present
    if re.search(r"\.\.\s*[a-zA-Z]+::", source_concat) is None:
        return

    # Separate content into lines (docstrings already are)
    if source_type == "File":
        source = source.split("\n")

    # Check for bad formatting
    for idx, line in enumerate(source):
        # Check for missing space after '..'
        missing = re.search(r"\.\.[a-zA-Z]+::", line)
        if missing is not None:
            sphinx_logger.warning(
                f"{source_type} '{name}' is missing a space after '..' in the "
                f"directive '{missing.group()}'"
            )
        # Extra spaces after '..' don't affect formatting

        # Check for missing preceding blank line
        # (exceptions are for directives at the start of files, after a header, or after
        # another directive/another directive's content)
        if idx == 0:
            continue
        dir_pattern = r"^\s*\.\. \w+::"  # line might start with whitespace
        head_pattern = r"^[-|=|\^]+$"
        directive = re.search(dir_pattern, line)
        if directive is not None:
            line_prev = source[idx - 1].strip()
            if (  # If previous line is...
                line_prev != ""  # not empty
                and not re.search(head_pattern, line_prev)  # not a header
                and not re.search(dir_pattern, line_prev)  # not a directive
            ):
                # Check if previous line is part of another directive
                bad = True
                for line_prev in reversed(source[: idx - 1]):
                    line_prev = line_prev.strip()
                    if line_prev == "" or re.search(head_pattern, line_prev):
                        # is a blank line or header, so not part of another directive
                        break  # must be bad formatting
                    if re.search(dir_pattern, line_prev):
                        bad = False  # is part of another directive, is good formatting
                        break
                    # or keep going until we reach the first line (so must be bad)
                if bad:
                    sphinx_logger.warning(
                        f"{source_type} '{name}' is missing a blank line before the "
                        f"directive '{directive.group()}' on line {idx + 1}"
                    )
