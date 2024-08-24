# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from docutils.nodes import reference
from docutils.parsers.rst.roles import set_classes


def gh_role(name, rawtext, text, lineno, inliner, options={}, content=[]):  # noqa: B006
    """Link to a GitHub issue.

    adapted from
    https://doughellmann.com/blog/2010/05/09/defining-custom-roles-in-sphinx/
    """
    try:
        # issue/PR mode (issues/PR-num will redirect to pull/PR-num)
        int(text)
    except ValueError:
        # direct link mode
        slug = text
    else:
        slug = "issues/" + text
    text = "#" + text
    ref = "https://github.com/mne-tools/mne-python/" + slug
    set_classes(options)
    node = reference(rawtext, text, refuri=ref, **options)
    return [node], []


def setup(app):
    app.add_role("gh", gh_role)
    return
