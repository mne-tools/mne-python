# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import docutils
from docutils.nodes import reference

# Adapted from sphinx
if docutils.__version_info__[:2] < (0, 22):
    from docutils.parsers.rst import roles

    def _normalize_options(options):
        if options is None:
            return {}
        n_options = options.copy()
        roles.set_classes(n_options)
        return n_options

else:
    from docutils.parsers.rst.roles import (
        normalize_options as _normalize_options,
    )


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
    options = _normalize_options(options)
    node = reference(rawtext, text, refuri=ref, **options)
    return [node], []


def setup(app):
    app.add_role("gh", gh_role)
    return
