# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from docutils import nodes


def unit_role(name, rawtext, text, lineno, inliner, options={}, content=[]):  # noqa: B006
    parts = text.split()

    def pass_error_to_sphinx(rawtext, text, lineno, inliner):
        msg = inliner.reporter.error(
            f"The :unit: role requires a space-separated number and unit; got {text}",
            line=lineno,
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    # ensure only two parts
    if len(parts) != 2:
        return pass_error_to_sphinx(rawtext, text, lineno, inliner)
    # ensure first part is a number
    try:
        _ = float(parts[0])
    except ValueError:
        return pass_error_to_sphinx(rawtext, text, lineno, inliner)
    # input is well-formatted: proceed
    node = nodes.Text("\u202f".join(parts))
    return [node], []


def setup(app):
    app.add_role("unit", unit_role)
    return
