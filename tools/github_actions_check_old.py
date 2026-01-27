#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import importlib
import re
import sys
from pathlib import Path

want_parts = 7  # should be updated when we add more pins!
regex = re.compile(r"^  - ([a-zA-Z\-]+) =([0-9.]+)$", re.MULTILINE)
this_root = Path(__file__).parent
env_old_text = (this_root / "environment_old.yml").read_text("utf-8")
parts = regex.findall(env_old_text)
assert len(parts) == want_parts, f"{len(parts)=} != {want_parts=}"
bad = list()
mod_name_map = {
    "scikit-learn": "sklearn",
}
for mod_name, want_ver in parts:
    if mod_name == "python":
        got_ver = ".".join(map(str, sys.version_info[:2]))
    else:
        mod = importlib.import_module(mod_name_map.get(mod_name, mod_name))
        got_ver = mod.__version__.lstrip("v")  # pooch prepends v
    if ".".join(got_ver.split(".")[:2]) != want_ver:
        bad.append(f"{mod_name}: {got_ver} != {want_ver}")
if bad:
    raise RuntimeError("At least one module is the wrong version:\n" + "\n".join(bad))
