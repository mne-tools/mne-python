# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re

from mne_doc_utils import sphinx_logger

# Recognised Sphinx directives (current as of v9.1.0)
DIRECTIVE_NAMES = [
    # Table of contents
    "toctree",
    # Admonitions, messages, warnings
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
    # Changes between versions
    "version-added",
    "versionadded",
    "versionchanged",
    "version-changed",
    "version-deprecated",
    "deprecated",
    "version-removed",
    "versionremoved",
    # Presentational
    "rubric",
    "centered",
    "hlist",
    # Code examples
    "highlight",
    "code-block",
    "sourcecode",
    "code",
    "literalinclude",
    # Glossary
    "glossary",
    # Meta-information
    "sectionauthor",
    "codeauthor",
    # Index-generating markup
    "index",
    # Including content
    "only",
    # Tables
    "tabularcolumns",
    # Math
    "math",
    # Grammar production
    "productionlist",
]


def setup(app):
    app.connect("source-read", check_directive_formatting)
    app.connect("autodoc-process-docstring", check_directive_formatting)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def check_directive_formatting(*args):
    """Check that directives are not malformed.

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

    # Note: we start the search with ^\s*, based on the assumption that directives
    # occur at the start of a line, possibly after whitespace

    # Check if text resembling directives is present
    if re.search(r"^\s*\.\.\s*[a-zA-Z\-]+\s*:", source_concat) is None:
        return

    # Separate content into lines (docstrings already are)
    if source_type == "File":
        source = source.split("\n")

    # Check for bad formatting
    for idx, line in enumerate(source):
        # Check for missing space after '..'
        missing = re.search(r"^\s*\.\.[a-zA-Z\-]+\s*:", line)
        if missing is not None:
            sphinx_logger.warning(
                f"{source_type} '{name}' is missing a space after '..' in the "
                f"directive '{missing.group()}'"
            )
        # Extra spaces after '..' don't affect formatting

        # Check for bad number of final colons (should be exactly 2)
        bad_colons = re.search(r"^\s*\.\.\s*([a-zA-Z\-]+)\s*(?<!:)(:{3,}|:)(?!:)", line)
        if bad_colons is not None:
            # Strip out directive name and check if it's a recognised directive
            # (links for files/sections take the same form, but are valid with a single
            # colon)
            directive_name = bad_colons.group(1)
            if directive_name in DIRECTIVE_NAMES:
                sphinx_logger.warning(
                    f"{source_type} '{name}' has bad number of final colons (i.e., not "
                    f"2) in the directive '{bad_colons.group()}'"
                )
        # Space(s) between directive name and final colons don't affect formatting

        # Check for missing preceding blank line
        # (exceptions are for directives at the start of files, after a header, or after
        # another directive/another directive's content)
        if idx == 0:
            continue
        dir_pattern = r"^\s*\.\.\s*[a-zA-Z\-]+\s*::"
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
