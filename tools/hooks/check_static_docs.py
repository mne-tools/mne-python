#!/usr/bin/env python
"""Check (or fix) statically filled docstrings against ``mne.utils.docs.docdict``.

Functions/methods decorated with ``@fill_doc_static(*keys)`` or
``@verbose_static(*keys)`` must contain, verbatim, the expanded text of
``docdict[key]`` for each key (``verbose_static`` implies the ``"verbose"`` key).
Methods decorated with ``@copy_doc_static(source)`` or
``@copy_function_doc_to_method_doc_static(source)`` must start with the
(transformed) docstring of ``source``. Unlike the dynamic decorators, nothing is
substituted at import time, so IDEs and other static tools see the complete
docstring.

Usage::

    python tools/hooks/check_static_docs.py [--fix] FILE [FILE ...]

Without ``--fix`` a non-zero exit status and a diff are emitted for every
docstring that is out of sync; with ``--fix`` the files are rewritten. Leftover
``%(key)s`` placeholders are expanded (and added to the decorator), which makes
migrating a function from ``@fill_doc`` a matter of renaming the decorator and
running ``--fix``.

How blocks are located (no fences are needed):

- A *parameter* entry (``name : type`` ...) is found by its parameter name and
  extends to the next line at the same or lower indentation.
- Any other entry is found by its first line and spans as many paragraphs as the
  ``docdict`` entry has. If the first line itself changed, the previous version of
  ``docdict`` (from ``git HEAD``) is used to find the old block.

Text specific to one docstring may follow a shared block (a ``.. versionadded::``
note, an extra sentence); the block's previous version (from ``git HEAD``) is used
to tell the two apart when the shared part changes. New lines added directly
after the shared text count as shared; start them with a blank line to mark them
as specific to the docstring.

Shared text may be edited either in ``mne/utils/docs.py`` or in one docstring: the
side that changed since ``git HEAD`` wins and is propagated to the other (with
``--fix``); edits that cannot be attributed to one side are rejected.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import argparse
import ast
import difflib
import functools
import importlib
import inspect
import re
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Make the repo's ``mne`` importable from the isolated pre-commit environment.
sys.path.insert(0, str(REPO))
from mne.utils.docs import _copy_doc, _copy_function_doc, docdict  # noqa: E402

_FILL_DECORATORS = {"fill_doc_static", "verbose_static"}
_COPY_DECORATORS = {"copy_doc_static", "copy_function_doc_to_method_doc_static"}
_PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")
_PARAM_RE = re.compile(r"^(\*{0,2}\w+(?:, \*{0,2}\w+)*)\s*:")
_LINE_LENGTH = 88  # ruff's default, which pyproject.toml does not override


class DocError(Exception):
    """Error in a statically documented function."""


# --------------------------------------------------------------------------
# docdict access


def _entry_lines(text):
    """Split a docdict entry into lines without surrounding blank lines."""
    return [line.rstrip() for line in text.strip("\n").splitlines()]


@functools.cache
def _old_docdict():
    """Return the docdict from ``git HEAD``, to locate blocks whose text changed."""
    try:
        source = subprocess.run(
            ["git", "show", "HEAD:mne/utils/docs.py"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    module = types.ModuleType("mne.utils._docs_head")
    module.__package__ = "mne.utils"
    try:
        exec(compile(source, "<HEAD:mne/utils/docs.py>", "exec"), module.__dict__)
    except Exception:
        return {}
    return dict(module.docdict)


def _iter_functions(tree):
    """Yield ``(qualname, node)`` for every function definition in ``tree``."""
    stack = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                qualname = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    yield qualname, child
                stack.append((child, qualname + "."))
            else:
                stack.append((child, prefix))


def _docstring_node(node):
    body0 = node.body[0] if node.body else None
    if (
        isinstance(body0, ast.Expr)
        and isinstance(body0.value, ast.Constant)
        and isinstance(body0.value.value, str)
    ):
        return body0.value
    return None


@functools.cache
def _old_bodies(path):
    """Return {qualname: docstring body} for ``path`` as of ``git HEAD``."""
    try:
        source = subprocess.run(
            ["git", "show", f"HEAD:{path.resolve().relative_to(REPO).as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        ).stdout
        tree = ast.parse(source)
    except (OSError, ValueError, subprocess.CalledProcessError, SyntaxError):
        return {}
    offset = _offsets(source)
    out = {}
    for qualname, node in _iter_functions(tree):
        const = _docstring_node(node)
        if const is None:
            continue
        start = offset(const.lineno, const.col_offset)
        stop = offset(const.end_lineno, const.end_col_offset)
        try:
            out[qualname] = _literal_parts(source[start:stop])[1]
        except DocError:
            pass
    return out


def _indent_of(line):
    return len(line) - len(line.lstrip())


def _paragraphs(lines, start):
    """Yield (start, stop) spans of consecutive non-blank lines from ``start``."""
    ii = start
    while ii < len(lines):
        if not lines[ii].strip():
            ii += 1
            continue
        stop = ii
        while stop < len(lines) and lines[stop].strip():
            stop += 1
        yield ii, stop
        ii = stop


def _n_paragraphs(lines):
    return sum(1 for _ in _paragraphs(lines, 0))


# --------------------------------------------------------------------------
# locating and replacing blocks


def _reindent(entry, indent):
    return [indent + line.rstrip() if line.strip() else "" for line in entry]


def _deindent(lines, indent):
    return [line[indent:].rstrip() if line.strip() else "" for line in lines]


def _block_end(lines, start, entry):
    """Return the end of the block starting at ``start`` for ``entry``."""
    if _PARAM_RE.match(entry[0]):  # structural: until the next line at <= indent
        indent = _indent_of(lines[start])
        stop = start + 1
        while stop < len(lines):
            line = lines[stop]
            if line.strip() and _indent_of(line) <= indent:
                break
            stop += 1
    else:  # prose: as many paragraphs as the entry has
        spans = list(_paragraphs(lines, start))[: _n_paragraphs(entry)]
        stop = spans[-1][1] if spans else start + 1
    while stop > start + 1 and not lines[stop - 1].strip():  # trailing blanks
        stop -= 1
    return stop


def _anchor_candidates(lines, entry, min_indent):
    """Return line indices where ``entry`` could start."""
    first = entry[0].strip()
    match = _PARAM_RE.match(first)
    anchor = match.group(1) if match else first
    out = []
    for ii, line in enumerate(lines):
        if _indent_of(line) < min_indent or not line.strip():
            continue
        stripped = line.strip()
        if match:
            this = _PARAM_RE.match(stripped)
            if this and this.group(1) == anchor:
                out.append(ii)
        elif stripped == anchor:
            out.append(ii)
    return out


def _score(lines, start, entry):
    """Count how many leading lines of ``entry`` match the docstring at ``start``."""
    indent = " " * _indent_of(lines[start])
    want = _reindent(entry, indent)
    got = lines[start : start + len(want)]
    return sum(1 for a, b in zip(want, got) if a.rstrip() == b.rstrip())


def _locate(lines, key, entry, min_indent):
    """Return ``(start, stop, version)`` of the block for ``entry`` in ``lines``.

    ``version`` is the entry text the block was found with: the current one, or
    the ``git HEAD`` one if the entry's anchor line changed.
    """
    versions = [entry]
    old = _old_docdict().get(key)
    if old is not None and _entry_lines(old) != entry:
        versions.append(_entry_lines(old))
    for version in versions:
        candidates = _anchor_candidates(lines, version, min_indent)
        if not candidates:
            continue
        scored = sorted((_score(lines, c, version), c) for c in candidates)
        best, start = scored[-1]
        if len(scored) > 1 and scored[-2][0] == best:
            raise DocError(
                f"ambiguous location for docdict[{key!r}]: lines "
                f"{[c + 1 for sc, c in scored if sc == best]} all start with "
                f"{version[0].strip()!r}"
            )
        return start, _block_end(lines, start, version), version
    raise DocError(
        f"could not find the block for docdict[{key!r}] (first line "
        f"{entry[0].strip()!r}); add it (``%({key})s`` on its own line) or paste it "
        "by hand"
    )


def _expand_placeholders(lines, keys, entries=None):
    """Expand ``%(key)s`` lines in place; return the keys found."""
    entries = docdict if entries is None else entries
    found = []
    out = []
    for line in lines:
        found_here = _PLACEHOLDER_RE.findall(line)
        if not found_here:
            out.append(line)
            continue
        if len(found_here) > 1 or not line.strip().startswith("%("):
            raise DocError(
                f"inline placeholder use {line.strip()!r} is not supported; "
                "write the text out by hand"
            )
        key = found_here[0]
        if key not in entries:
            raise DocError(f"unknown docdict key {key!r}")
        indent = " " * _indent_of(line)
        entry = _entry_lines(entries[key])
        out.extend(_reindent(entry, indent))
        suffix = line.split(")s", 1)[1].strip()
        if suffix:  # trailing text moves to its own line, at the entry's indent
            out.append(" " * _indent_of(out[-1]) + suffix)
        found.append(key)
    lines[:] = out
    return found


def _old_block_info(old_body, key, entry, old_entry):
    """Return ``(old own text, first line after the old block)`` from ``git HEAD``.

    The own text is what followed the shared text *inside* the old block; the
    line after the block bounds it from below when the block's extent cannot be
    determined from its content. Returns ``None`` if the old docstring (or the
    block in it) cannot be found.
    """
    if old_body is None:
        return None
    old_lines = old_body.splitlines()
    widths = [_indent_of(x) for x in old_lines[1:] if x.strip()]
    try:
        # the old docstring may predate its migration and still hold placeholders
        _expand_placeholders(old_lines, [], _old_docdict() or docdict)
        # the old body holds the *old* text, so locate with the old entry's extent
        look_for = old_entry if old_entry is not None else entry
        start, stop, _ = _locate(old_lines, key, look_for, min(widths) if widths else 0)
    except DocError:
        return None
    block = _deindent(old_lines[start:stop], _indent_of(old_lines[start]))
    old_next = next((x.strip() for x in old_lines[stop:] if x.strip()), None)
    if old_next is not None and old_next.startswith(('"', "'")):
        old_next = None  # the closing quotes: the block ended the docstring
    for known in (old_entry, entry):
        if known is not None and block[: len(known)] == known:
            return block[len(known) :], old_next
    return None


def _bounded_stop(lines, start, old_next):
    """End of the block at ``start``, bounded by the line that used to follow it."""
    if old_next is None:  # the old block ran to the end of the docstring
        stop = len(lines)
    else:
        stop = next(
            (
                ii
                for ii in range(start + 1, len(lines))
                if lines[ii].strip() == old_next
            ),
            None,
        )
        if stop is None:
            return None
    while stop > start + 1 and not lines[stop - 1].strip():
        stop -= 1
    return stop


def _split_block(current, entry, old_entry, old_own):
    """Split a block into (shared text, site-specific text that follows it)."""
    shared, own = _split_block_raw(current, entry, old_entry, old_own)
    while shared and not shared[-1].strip():  # a separator belongs to the own text
        own.insert(0, shared.pop())
    return shared, own


def _split_block_raw(current, entry, old_entry, old_own):
    """Split a block without normalizing blank lines at the boundary.

    ``old_own`` is the site-specific text as of ``git HEAD`` (``None`` if
    unknown). Lines added directly after the shared text count as shared unless
    they start with a blank line, which marks a site-specific addition (as in
    the ``.. versionadded::`` notes that follow many parameters).
    """
    # prefer the longest matching version: a verbatim match of a longer (old)
    # entry outweighs a shorter match followed by what looks like own text
    knowns = [k for k in (entry, old_entry) if k is not None]
    knowns = sorted({tuple(k): list(k) for k in knowns}.values(), key=len, reverse=True)
    if old_own is None:  # no previous version of this docstring (not in git)
        for known in knowns:
            if current[: len(known)] == known:
                return list(current[: len(known)]), list(current[len(known) :])
        if len(current) == len(entry):  # edited in place, nothing follows it
            return list(current), []
        raise DocError(
            "the shared text changed, but its previous version is not available "
            "(not in git HEAD) to tell it apart from the text following it; make "
            "the change in mne/utils/docs.py instead"
        )
    if old_own and current[-len(old_own) :] != old_own:
        # the site-specific text was edited, so the shared text must be intact
        for known in knowns:
            if current[: len(known)] == known:
                return list(current[: len(known)]), list(current[len(known) :])
        raise DocError(
            "both the shared text and the text following it changed in this "
            "docstring; make the shared change in mne/utils/docs.py instead"
        )
    head = current[: len(current) - len(old_own)] if old_own else list(current)
    for known in knowns:
        if head[: len(known)] != known:
            continue
        rest = head[len(known) :]
        if not rest:  # the whole head is this version of the shared text
            return list(known), list(old_own)
        if rest[:1] == [""]:  # a blank line marks a site-specific addition
            return list(known), list(rest) + list(old_own)
    return list(head), list(old_own)


def expected_fill(body, keys, reverse, old_body=None):
    """Return (new body, all keys) with every entry matching ``docdict``.

    If an entry is unchanged since ``git HEAD`` but the docstring's copy of it was
    edited, the edit is recorded in ``reverse`` (key -> new text) to be pushed
    back into ``docdict`` rather than overwritten. ``old_body`` is this
    docstring as of ``git HEAD``, used to tell shared text apart from the
    site-specific text that may follow it.
    """
    lines = body.splitlines()
    widths = [_indent_of(x) for x in lines[1:] if x.strip()]
    min_indent = min(widths) if widths else 0
    keys = list(keys)
    for key in _expand_placeholders(lines, keys):
        if key not in keys:
            keys.append(key)
    for key in keys:
        if key not in docdict:
            raise DocError(f"unknown docdict key {key!r}")
        entry = _entry_lines(docdict[key])
        start, stop, _ = _locate(lines, key, entry, min_indent)
        indent = _indent_of(lines[start])
        current = _deindent(lines[start:stop], indent)
        old_entry = _old_docdict().get(key)
        old_entry = _entry_lines(old_entry) if old_entry is not None else None
        old_info = _old_block_info(old_body, key, entry, old_entry)
        old_own = old_info[0] if old_info is not None else None
        knowns = [k for k in (entry, old_entry) if k is not None]
        if old_info is not None and old_entry is not None and old_entry != entry:
            # a stale copy may extend past the new entry's extent (e.g. the
            # entry lost a paragraph); recognize it by the old entry's full text
            bounded = _bounded_stop(lines, start, old_info[1])
            if bounded is not None and bounded > stop:
                extended = _deindent(lines[start:bounded], indent)
                if extended[: len(old_entry)] == old_entry:
                    stop, current = bounded, extended
        if old_info is not None and not any(current[: len(k)] == k for k in knowns):
            # The block content matches no known version of the entry, so its
            # extent cannot be trusted either; bound it by the line that
            # followed the old block instead.
            bounded = _bounded_stop(lines, start, old_info[1])
            if bounded is None:
                raise DocError(
                    f"docdict[{key!r}]: cannot tell where the edited shared text "
                    "ends (the text that used to follow it is gone); make the "
                    "change in mne/utils/docs.py instead"
                )
            stop = bounded
            current = _deindent(lines[start:stop], indent)
        try:
            shared, own = _split_block(current, entry, old_entry, old_own)
        except DocError as exc:
            raise DocError(f"docdict[{key!r}]: {exc}") from None
        if shared == entry:
            continue  # in sync; anything after the entry is the site's own text
        if old_entry == entry:
            # docdict is unchanged, so the docstring's copy is what was edited
            reverse[key] = shared
            continue
        lines[start:stop] = _reindent(entry + own, " " * indent)
    new = "\n".join(lines)
    if body.endswith("\n"):
        new += "\n"
    return new, keys


def _resolve(source):
    """Resolve a ``"func:mne.viz.plot_raw"`` / ``"meth:mne.io.Raw.plot"`` source."""
    kind, _, path = source.partition(":")
    if kind not in ("func", "meth") or not path:
        raise DocError(f"copy source must look like 'func:some.path', got {source!r}")
    module_name, _, attr = path.rpartition(".")
    if kind == "meth":
        module_name, _, cls_name = module_name.rpartition(".")
    # reload so that a source docstring fixed earlier in this run is what we copy
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if kind == "meth":
        obj = getattr(getattr(module, cls_name), attr)
    else:
        obj = getattr(module, attr)
    return kind, obj


def expected_copy(body, source, default_indent):
    """Return the body for a method whose docstring is copied from ``source``."""
    kind, obj = _resolve(source)
    dummy = types.SimpleNamespace(__doc__=None)
    (_copy_function_doc if kind == "func" else _copy_doc)(obj, dummy)
    expected = inspect.cleandoc(dummy.__doc__).splitlines()
    lines = body.splitlines()
    widths = [_indent_of(x) for x in lines[1:] if x.strip()]
    indent = " " * min(widths) if widths else default_indent
    current = []
    if body.strip() not in ("", "."):
        current = inspect.cleandoc(body).splitlines()
    # the method's own text (if any) follows the copied part
    n_match = sum(1 for a, b in zip(expected, current) if a.rstrip() == b.rstrip())
    if n_match == len(expected):
        own = current[len(expected) :]
    else:  # copied part changed: assume it still spans the same paragraphs
        spans = list(_paragraphs(current, 0))[: _n_paragraphs(expected)]
        own = current[spans[-1][1] :] if spans else []
    new = expected + own
    while new and not new[-1].strip():
        new.pop()
    new = [new[0]] + _reindent(new[1:], indent) + [indent]
    return "\n".join(new)


# --------------------------------------------------------------------------
# source editing


def _literal_parts(segment):
    """Split a string-literal source segment into (prefix, body, quote)."""
    m = re.match(r"^([rRuUbBfF]*)(\"\"\"|'''|\"|')", segment)
    if m is None or not segment.endswith(m.group(2)):
        raise DocError("could not parse docstring literal")
    quote = m.group(2)
    return m.group(1), segment[m.end() : -len(quote)], quote


def _decorator(node):
    """Return (kind, call node, arguments) for a static-doc decorator, if any."""
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _FILL_DECORATORS | _COPY_DECORATORS:
            continue
        args = []
        for arg in dec.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                raise DocError(f"@{name} arguments must be string literals")
            args.append(arg.value)
        if name in _COPY_DECORATORS and len(args) != 1:
            raise DocError(f"@{name} takes exactly one source string")
        return name, dec, args
    return None


def _offsets(source):
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset(lineno, col):  # ast columns are in UTF-8 bytes
        start = line_starts[lineno - 1]
        return start + len(source[start:].encode()[:col].decode())

    return offset


def process_file(path, fix, reverse, *, kinds=_FILL_DECORATORS | _COPY_DECORATORS):
    """Check one file; return a list of error messages."""
    source = path.read_text()
    offset = _offsets(source)
    errors = []
    edits = []  # (start, stop, replacement)
    rerun = False  # a docstring was inserted and still needs filling
    for qualname, node in _iter_functions(ast.parse(source)):
        where = f"{path}:{node.lineno} {node.name}"
        try:
            found = _decorator(node)
            if found is None:
                continue
            name, dec, args = found
            if name not in kinds:
                continue
            body0 = node.body[0] if node.body else None
            const = _docstring_node(node)
            if const is None:
                if fix and name in _COPY_DECORATORS and body0 is not None:
                    at = offset(body0.lineno, 0)
                    edits.append((at, at, " " * body0.col_offset + '"""."""\n'))
                    rerun = True
                    continue
                raise DocError('needs a docstring (use """.""" for a pure copy)')
            start, stop = (
                offset(const.lineno, const.col_offset),
                offset(const.end_lineno, const.end_col_offset),
            )
            prefix, body, quote = _literal_parts(source[start:stop])
            if name in _COPY_DECORATORS:
                want = expected_copy(body, args[0], " " * const.col_offset)
                new_args = args
            else:
                implied = ["verbose"] if name == "verbose_static" else []
                old_body = _old_bodies(path).get(qualname)
                want, keys = expected_fill(body, implied + args, reverse, old_body)
                new_args = [k for k in keys if k not in implied]
        except DocError as exc:
            errors.append(f"{where}: {exc}")
            continue
        except Exception as exc:  # e.g. import error resolving a copy source
            errors.append(f"{where}: {type(exc).__name__}: {exc}")
            continue
        # overlong shared text gets the usual E501 suppression comment on the
        # closing quotes (reflowing the docdict entry is the nicer fix when practical)
        rest_of_line = source[stop:].split("\n", 1)[0]
        noqa_edit = None
        overlong = any(len(line) > _LINE_LENGTH for line in want.splitlines())
        if overlong and not rest_of_line.strip():
            noqa_edit = (stop, stop, "  # noqa: E501")
        elif not overlong and rest_of_line == "  # noqa: E501":
            noqa_edit = (stop, stop + len(rest_of_line), "")  # no longer needed
        if want == body and new_args == args and noqa_edit is None:
            continue
        if "\\" in want and "r" not in prefix.lower():
            if "\\" in body and body != want:
                errors.append(
                    f"{where}: expanded text contains a backslash; make this a raw "
                    "(r) docstring first"
                )
                continue
            prefix_edit = (start, start + len(prefix), "r" + prefix)
        else:
            prefix_edit = None
        if fix:
            body_start = start + len(prefix) + len(quote)
            edits.append((body_start, stop - len(quote), want))
            if noqa_edit is not None:
                edits.append(noqa_edit)
            if prefix_edit is not None:
                edits.append(prefix_edit)
            if new_args != args:
                dec_start, dec_stop = (
                    offset(dec.lineno, dec.col_offset),
                    offset(dec.end_lineno, dec.end_col_offset),
                )
                call = f"{name}({', '.join(repr(a) for a in new_args)})".replace(
                    "'", '"'
                )
                edits.append((dec_start, dec_stop, call))
        elif want == body:  # only the E501 suppression comment is stale
            verb = "needs" if noqa_edit[2] else "no longer needs"
            errors.append(
                f"{where}: docstring {verb} a trailing ``# noqa: E501``; run\n"
                f"    python tools/hooks/check_static_docs.py --fix {path}"
            )
        else:
            diff = difflib.unified_diff(
                body.splitlines(keepends=True),
                want.splitlines(keepends=True),
                "current docstring",
                "expected",
            )
            what = (
                f"copied docstring of {args[0]!r}"
                if name in _COPY_DECORATORS
                else "mne/utils/docs.py::docdict"
            )
            errors.append(
                f"{where}: docstring out of sync with {what}.\n"
                "  Shared text must be edited at its source (not in this docstring); "
                "then run\n"
                f"    python tools/hooks/check_static_docs.py --fix {path}\n"
                + "".join("  " + line for line in diff)
            )
    if edits:
        for start, stop, rep in sorted(edits, reverse=True):
            source = source[:start] + rep + source[stop:]
        path.write_text(source)
        print(f"fixed {len(edits)} docstring edit(s) in {path}")
        if rerun:
            errors.extend(process_file(path, fix, reverse))
    return errors


DOCS_PY = REPO / "mne" / "utils" / "docs.py"


def _write_docdict_entries(reverse):
    """Write edited entries back into docs.py; return (written keys, errors)."""
    source = DOCS_PY.read_text()
    offset = _offsets(source)
    edits, written, errors = [], [], []
    literals = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and getattr(target.value, "id", "") == "docdict"
            and isinstance(target.slice, ast.Constant)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            literals[target.slice.value] = node.value
    for key, new_lines in reverse.items():
        const = literals.get(key)
        if const is None or const.value != docdict[key]:
            errors.append(
                f"docdict[{key!r}] was edited in a docstring, but it is not a plain "
                "string literal in mne/utils/docs.py; edit it there by hand"
            )
            continue
        old = docdict[key]
        lead = "\n" if old.startswith("\n") else ""
        trail = "\n" if old.endswith("\n") else ""
        new = lead + "\n".join(new_lines) + trail
        start = offset(const.lineno, const.col_offset)
        stop = offset(const.end_lineno, const.end_col_offset)
        prefix, _, quote = _literal_parts(source[start:stop])
        if "\\" in new and "r" not in prefix.lower():
            prefix = "r" + prefix
        edits.append((start, stop, f"{prefix}{quote}{new}{quote}"))
        dict.__setitem__(docdict, key, new)  # BunchConst forbids reassignment
        written.append(key)
    for start, stop, rep_ in sorted(edits, reverse=True):
        source = source[:start] + rep_ + source[stop:]
    if edits:
        DOCS_PY.write_text(source)
        print(f"updated docdict[{', '.join(map(repr, written))}] in {DOCS_PY}")
    return written, errors


def _files_using(keys):
    """Return the files whose static decorators use ``keys`` (or copies, if empty)."""
    out = []
    for path in sorted((REPO / "mne").rglob("*.py")):
        if "tests" in path.parts:
            continue  # decorators in tests are examples, not documentation
        text = path.read_text()
        if "_static(" not in text:
            continue
        if (
            any(f'"{key}"' in text for key in keys)
            or ("verbose" in keys and "verbose_static(" in text)
            or (not keys and any(f"{d}(" in text for d in _COPY_DECORATORS))
        ):
            out.append(path)
    return out


def main(argv=None):
    """Run the check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="rewrite files in place")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    errors = []
    reverse = {}
    files = [p for p in args.files if p.suffix == ".py" and "_static(" in p.read_text()]
    # fill sites first, so that copies of them (processed second) see fixed text
    for kinds in (_FILL_DECORATORS, _COPY_DECORATORS):
        for path in files:
            errors.extend(process_file(path, args.fix, reverse, kinds=kinds))
    if reverse and not args.fix:
        for key in reverse:
            errors.append(
                f"docdict[{key!r}] was edited in a docstring; run with --fix to push "
                "the edit into mne/utils/docs.py and every other docstring using it"
            )
    elif reverse:
        written, rev_errors = _write_docdict_entries(reverse)
        errors.extend(rev_errors)
        for path in _files_using(written):
            errors.extend(process_file(path, True, {}, kinds=_FILL_DECORATORS))
        for path in _files_using([]):  # copies of anything just updated
            errors.extend(process_file(path, True, {}, kinds=_COPY_DECORATORS))
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
