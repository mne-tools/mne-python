# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import sys
from importlib import import_module
from pathlib import Path

import ast_comments as ast
import black

import mne


class RewriteAssign(ast.NodeTransformer):
    """NodeTransformer to replace lazy attach with attach_stub."""

    def visit_Assign(self, node):
        """Replace lazy attach assignment with stub assignment."""
        if not hasattr(node.targets[0], "dims"):
            return node

        ids = [name.id for name in node.targets[0].dims]
        if ids == ["__getattr__", "__dir__", "__all__"]:
            return ast.parse(
                "__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)\n"
            )
        return node


pyi_mode = black.Mode(is_pyi=True)
root = Path(mne.__file__).parent
inits = root.rglob("__init__.py")

for init in inits:
    # skip init files that don't lazy load (e.g., tests)
    code = init.read_text("utf-8")
    if "import lazy_loader as lazy" not in code:
        continue
    # get the AST
    tree = ast.parse(code)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(nodes) == 1
    node = nodes[0]
    keywords = node.value.keywords
    # get submodules
    import_lines = list()
    assert keywords[0].arg == "submodules"
    # for submod in keywords[0].value.elts:
    #     import_lines.append(f"import {submod.value}")
    submods = [submod.value for submod in keywords[0].value.elts]
    if len(submods):
        import_lines.append(f"from . import {', '.join(submods)}")
    # get attrs
    assert keywords[1].arg == "submod_attrs"
    _dict = keywords[1].value
    for key, vals in zip(_dict.keys, _dict.values):
        attrs = [attr.value for attr in vals.elts]
        import_lines.append(f"from .{key.value} import {', '.join(attrs)}")
    # format
    import_lines = black.format_str("\n".join(import_lines), mode=pyi_mode)
    # get __all__
    import_path = str(init.parent.relative_to(root.parent)).replace(os.sep, ".")
    import_module(import_path)
    _all = black.format_str(
        f"__all__ = {repr(sys.modules[import_path].__all__)}\n",
        mode=pyi_mode,
    )
    # write __init__.pyi
    outfile = init.with_suffix(".pyi")
    with open(outfile, "w") as fid:
        fid.write(_all)
        fid.write(import_lines)
    # rewrite __init__.py
    new_tree = RewriteAssign().visit(tree)
    new_tree = ast.fix_missing_locations(new_tree)
    new_code = ast.unparse(new_tree)
    formatted_code = black.format_str(new_code, mode=black.Mode())
    with open(init, "w") as fid:
        fid.write(formatted_code)
