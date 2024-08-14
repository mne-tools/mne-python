# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from docutils.nodes import reference, strong, target


def newcontrib_role(name, rawtext, text, lineno, inliner, options={}, content=[]):  # noqa: B006
    """Create a role to highlight new contributors in changelog entries."""
    newcontrib = f"new contributor {text}"
    alias_text = f" <{text}_>"
    rawtext = f"`{newcontrib}{alias_text}`_"
    refname = text.lower()
    strong_node = strong(rawtext, newcontrib)
    target_node = target(alias_text, refname=refname, names=[newcontrib])
    target_node.indirect_reference_name = text
    options.update(refname=refname, name=newcontrib)
    ref_node = reference("", "", strong_node, **options)
    ref_node[0].rawsource = rawtext
    inliner.document.note_indirect_target(target_node)
    inliner.document.note_refname(ref_node)
    return [ref_node, target_node], []


def setup(app):
    app.add_role("newcontrib", newcontrib_role)
    return
