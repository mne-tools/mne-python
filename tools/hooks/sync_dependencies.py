#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import difflib
import re

# NB here we use metadata from the latest stable release because this goes in our
# README, which should apply to the latest release (rather than dev).
# For oldest supported dev dependencies, see update_environment_file.py.
from importlib.metadata import metadata
from pathlib import Path

from mne.utils import _pl, warn

README_PATH = Path(__file__).parents[2] / "README.rst"
BEGIN = ".. ↓↓↓ BEGIN CORE DEPS LIST. DO NOT EDIT! HANDLED BY PRE-COMMIT HOOK ↓↓↓"
END = ".. ↑↑↑ END CORE DEPS LIST. DO NOT EDIT! HANDLED BY PRE-COMMIT HOOK ↑↑↑"

CORE_DEPS_URLS = {
    "Python": "https://www.python.org",
    "NumPy": "https://numpy.org",
    "SciPy": "https://scipy.org",
    "Matplotlib": "https://matplotlib.org",
    "Pooch": "https://www.fatiando.org/pooch/latest/",
    "tqdm": "https://tqdm.github.io",
    "Jinja2": "https://palletsprojects.com/p/jinja/",
    "decorator": "https://github.com/micheles/decorator",
    "lazy-loader": "https://pypi.org/project/lazy_loader",
    "packaging": "https://packaging.pypa.io/en/stable/",
}


def _prettify_pin(pin):
    if pin is None:
        return ""
    pins = pin.split(",")
    replacements = {
        "<=": " ≤ ",
        ">=": " ≥ ",
        "<": " < ",
        ">": " > ",
    }
    for old, new in replacements.items():
        pins = [p.replace(old, new) for p in pins]
    pins = reversed(pins)
    return ",".join(pins)


# get the dependency info
py_pin = metadata("mne").get("Requires-Python")
all_deps = metadata("mne").get_all("Requires-Dist")
core_deps = [f"python{py_pin}", *[dep for dep in all_deps if "extra ==" not in dep]]
pattern = re.compile(r"(?P<name>[A-Za-z_\-\d]+)(?P<pin>[<>=]+.*)?")
core_deps_pins = {
    dep["name"]: _prettify_pin(dep["pin"]) for dep in map(pattern.match, core_deps)
}
# don't show upper pin on NumPy (not important for users, just devs)
new_pin = core_deps_pins["numpy"].split(",")
new_pin.remove(" < 3")
core_deps_pins["numpy"] = new_pin[0]

# make sure our URLs dict is minimal and complete
missing_urls = set(core_deps_pins) - {dep.lower() for dep in CORE_DEPS_URLS}
extra_urls = {dep.lower() for dep in CORE_DEPS_URLS} - set(core_deps_pins)
update_msg = (
    "please update `CORE_DEPS_URLS` mapping in `tools/hooks/sync_dependencies.py`."
)
if missing_urls:
    _s = _pl(missing_urls)
    raise RuntimeError(
        f"Missing URL{_s} for package{_s} {', '.join(missing_urls)}; {update_msg}"
    )
if extra_urls:
    _s = _pl(extra_urls)
    warn(f"Superfluous URL{_s} for package{_s} {', '.join(extra_urls)}; {update_msg}")

# construct the rST
core_deps_bullets = [
    f"- `{key} <{url}>`__{core_deps_pins[key.lower()]}"
    for key, url in CORE_DEPS_URLS.items()
]

# rewrite the README file
lines = README_PATH.read_text("utf-8").splitlines()
out_lines = list()
skip = False
for line in lines:
    if line.strip() == BEGIN:
        skip = True
        out_lines.append(line)
        out_lines.extend(["", *core_deps_bullets, ""])
    if line.strip() == END:
        skip = False
    if not skip:
        out_lines.append(line)
new = "\n".join(out_lines) + "\n"
old = README_PATH.read_text("utf-8")
if new != old:
    diff = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines()))
    print(f"Updating {README_PATH} with diff:\n{diff}")
    README_PATH.write_text(new, encoding="utf-8")
