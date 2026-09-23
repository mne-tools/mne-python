#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
import unicodedata
from pathlib import Path


def sort_key(line):
    """Sort ignoring case and accents, so "Sébastien" lands next to "Sebastian".

    A plain sort puts every accented character after "z", which exiles names
    like "Théodore" after "Thucydides".
    """
    decomposed = unicodedata.normalize("NFKD", line)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


if __name__ == "__main__":
    for path in map(Path, sys.argv[1:]):
        lines = path.read_text("utf-8").splitlines()
        ordered = sorted(lines, key=sort_key)
        if ordered != lines:
            path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
            print(f"Sorted {path}")
