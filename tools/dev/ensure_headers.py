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
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from pathlib import Path

import numpy as np
from git import Repo

repo = Repo(Path(__file__).parents[2])

LICENSE_LINE = "# License: BSD-3-Clause"
COPYRIGHT_LINE = "# Copyright the MNE-Python contributors."


def get_paths_from_tree(root, level=0):
    for entry in root:
        if entry.type == "tree":
            for x in get_paths_from_tree(entry, level + 1):
                yield x
        else:
            yield Path(entry.path)  # entry.type


def _ensure_license(lines, path):
    # 1. Keep existing
    license_idx = np.where([line.startswith("# License: ") for line in lines])[0]
    assert len(license_idx) <= 1, len(license_idx)
    if len(license_idx):  # If existing, ensure it's correct
        lines[license_idx[0]] = LICENSE_LINE
        return
    # 2. First non-comment line after author line
    author_idx = np.where([re.match(r"^# Authors? ?: .*$", line) for line in lines])[0]
    assert len(author_idx) <= 1, len(author_idx)
    if len(author_idx):
        insert = author_idx[0]
        for extra in range(1, 100):
            if not lines[insert + extra].startswith("#"):
                break
        else:
            raise RuntimeError(
                "Failed to find non-comment line within 100 of end of author line"
            )
        lines.insert(insert + extra, LICENSE_LINE)
        return
    # 3. First line after docstring
    insert = 0
    max_len = 100
    if lines[0].startswith('"""'):
        if lines[0].count('"""') != 2:
            for insert in range(1, max_len):
                if '"""' in lines[insert]:
                    # Find next non-blank line:
                    for extra in range(1, 3):  # up to 2 blank lines
                        if lines[insert + extra].strip():
                            break
                    else:
                        raise RuntimeError(
                            "Failed to find non-blank line within 2 of end of "
                            f"docstring at line {insert + 1}"
                        )
                    insert += extra
                    break
            else:
                raise RuntimeError(
                    f"Failed to find end of file docstring within {max_len} lines"
                )
        lines.insert(insert, LICENSE_LINE)
        return
    # 4. First non-comment line
    for insert in range(100):
        if not lines[insert].startswith("#"):
            lines.insert(insert, LICENSE_LINE)
            return
    else:
        raise RuntimeError("Failed to find non-comment line within 100 lines")


def _ensure_copyright(lines, path):
    n_expected = {
        "mne/preprocessing/_csd.py": 2,
        "mne/transforms.py": 2,
    }
    n_copyright = sum(line.startswith("# Copyright ") for line in lines)
    assert n_copyright <= n_expected.get(str(path), 1), n_copyright
    insert = lines.index(LICENSE_LINE) + 1
    if lines[insert].startswith("# Copyright"):
        lines[insert] = COPYRIGHT_LINE
    else:
        lines.insert(insert, COPYRIGHT_LINE)
    assert lines.count(COPYRIGHT_LINE) == 1, lines.count(COPYRIGHT_LINE)


for path in get_paths_from_tree(repo.tree()):
    if not path.suffix == ".py":
        continue
    lines = path.read_text("utf-8").split("\n")
    # Remove the UTF-8 file coding stuff
    orig_lines = list(lines)
    if lines[0] == "# -*- coding: utf-8 -*-":
        lines = lines[1:]
    _ensure_license(lines, path)
    _ensure_copyright(lines, path)
    if lines != orig_lines:
        print(path)
        path.write_text("\n".join(lines), "utf-8")
