"""Ensure license and copyright statements are in source files.

From https://www.bestpractices.dev/en/projects/7783?criteria_level=2:

    The project MUST include a copyright statement in each source file, identifying the
    copyright holder (e.g., the [project name] contributors). [copyright_per_file]
    This MAY be done by including the following inside a comment near the beginning of
    each file: "Copyright the [project name] contributors.".

And:

    The project MUST include a license statement in each source file.

This script ensures that we use consistent license naming in consistent locations
toward the top of each file.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from pathlib import Path

import numpy as np
from git import Repo

repo = Repo(Path(__file__).parents[2])

AUTHOR_LINE = "# Authors: The MNE-Python contributors."
LICENSE_LINE = "# License: BSD-3-Clause"
COPYRIGHT_LINE = "# Copyright the MNE-Python contributors."

# Cover how lines can start (regex or tuple to be used with startswith)
AUTHOR_RE = re.compile(r"^# (A|@a)uthors? ?: .*$")
LICENSE_STARTS = ("# License: ", "# SPDX-License-Identifier: ")
COPYRIGHT_STARTS = ("# Copyright ",)


def get_paths_from_tree(root, level=0):
    """Get paths from a GitPython tree."""
    for entry in root:
        if entry.type == "tree":
            yield from get_paths_from_tree(entry, level + 1)
        else:
            yield Path(entry.path)  # entry.type


def first_commentable_line(lines):
    """Find the first line where we can add a comment."""
    max_len = 100
    if lines[0].startswith(('"""', 'r"""')):
        if lines[0].count('"""') == 2:
            return 1
        for insert in range(1, min(max_len, len(lines))):
            if '"""' in lines[insert]:
                return insert + 1
        else:
            raise RuntimeError(
                f"Failed to find end of file docstring within {max_len} lines"
            )
    if lines[0].startswith("#!"):
        return 1
    else:
        return 0


def path_multi_author(path):
    """Check if a file allows multi-author comments."""
    return path.parts[0] in ("examples", "tutorials")


def get_author_idx(path, lines):
    """Get the index of the author line, if available."""
    author_idx = np.where([AUTHOR_RE.match(line) is not None for line in lines])[0]
    assert len(author_idx) <= 1, f"{len(author_idx)=} for {path=}"
    return author_idx[0] if len(author_idx) else None


def get_license_idx(path, lines):
    """Get the license index."""
    license_idx = np.where([line.startswith(LICENSE_STARTS) for line in lines])[0]
    assert len(license_idx) <= 1, f"{len(license_idx)=} for {path=}"
    return license_idx[0] if len(license_idx) else None


def _ensure_author(lines, path):
    author_idx = get_author_idx(path, lines)
    license_idx = get_license_idx(path, lines)
    first_idx = first_commentable_line(lines)
    # 1. Keep existing
    if author_idx is not None:
        # We have to be careful here -- examples and tutorials are allowed multiple
        # authors
        if path_multi_author(path):
            # Just assume it's correct and return
            return
        assert license_idx is not None, f"{license_idx=} for {path=}"
        for _ in range(license_idx - author_idx - 1):
            lines.pop(author_idx + 1)
        assert lines[author_idx + 1].startswith(LICENSE_STARTS), lines[license_idx + 1]
        del license_idx
        lines[author_idx] = AUTHOR_LINE
    elif license_idx is not None:
        # 2. Before license line if present
        lines.insert(license_idx, AUTHOR_LINE)
    else:
        # 3. First line after docstring
        lines.insert(first_idx, AUTHOR_LINE)
    # Now make sure it's in the right spot
    author_idx = get_author_idx(path, lines)
    if author_idx != 0:
        if author_idx == first_idx:
            # Insert a blank line
            lines.insert(author_idx, "")
            author_idx += 1
        first_idx += 1
    if author_idx != first_idx:
        raise RuntimeError(
            "\nLine should have comments as docstring or author line needs to be moved "
            "manually to be one blank line after the docstring:\n"
            f"{path}: {author_idx=} != {first_idx=}"
        )


def _ensure_license(lines, path):
    # 1. Keep/replace existing
    insert = get_license_idx(path, lines)

    # 2. After author line(s)
    if insert is None:
        author_idx = get_author_idx(path, lines)
        assert author_idx is not None, f"{author_idx=} for {path=}"
        insert = author_idx + 1
        if path_multi_author:
            # Figure out where to insert the license:
            for insert, line in enumerate(lines[author_idx + 1 :], insert):
                if not line.startswith("#     "):
                    break
    if lines[insert].startswith(LICENSE_STARTS):
        lines[insert] = LICENSE_LINE
    else:
        lines.insert(insert, LICENSE_LINE)
    assert lines.count(LICENSE_LINE) == 1, f"{lines.count(LICENSE_LINE)=} for {path=}"


def _ensure_copyright(lines, path):
    n_expected = {
        "mne/preprocessing/_csd.py": 2,
        "mne/transforms.py": 2,
    }
    n_copyright = sum(line.startswith(COPYRIGHT_STARTS) for line in lines)
    assert n_copyright <= n_expected.get(str(path), 1), n_copyright
    insert = lines.index(LICENSE_LINE) + 1
    if lines[insert].startswith(COPYRIGHT_STARTS):
        lines[insert] = COPYRIGHT_LINE
    else:
        lines.insert(insert, COPYRIGHT_LINE)
    assert lines.count(COPYRIGHT_LINE) == 1, (
        f"{lines.count(COPYRIGHT_LINE)=} for {path=}"
    )


def _ensure_blank(lines, path):
    assert lines.count(COPYRIGHT_LINE) == 1, (
        f"{lines.count(COPYRIGHT_LINE)=} for {path=}"
    )
    insert = lines.index(COPYRIGHT_LINE) + 1
    if lines[insert].strip():  # actually has content
        lines.insert(insert, "")


for path in get_paths_from_tree(repo.tree()):
    if not path.suffix == ".py":
        continue
    lines = path.read_text("utf-8").split("\n")
    # Remove the UTF-8 file coding stuff
    orig_lines = list(lines)
    if lines[0] in ("# -*- coding: utf-8 -*-", "# -*- coding: UTF-8 -*-"):
        lines = lines[1:]
        if lines[0] == "":
            lines = lines[1:]
    # We had these with mne/commands without an executable bit, and don't really
    # need them executable, so let's get rid of the line.
    if lines[0].startswith("#!/usr/bin/env python") and path.parts[:2] == (
        "mne",
        "commands",
    ):
        lines = lines[1:]
    _ensure_author(lines, path)
    _ensure_license(lines, path)
    _ensure_copyright(lines, path)
    _ensure_blank(lines, path)
    if lines != orig_lines:
        print(path)
        path.write_text("\n".join(lines), "utf-8")
