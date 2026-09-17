"""Specs to add to Pyodide's lock for JupyterLite, versus what it already curates.

Naming a package in ``PyodideLockAddon.specs`` drops its existing pin from
the lock (see ``pyodide_lock.uv_pip_compile.constraints_txt``) and resolves
it fresh from PyPI instead, a smaller and differently versioned set than
Pyodide's own curated wheels. :func:`jupyterlite_specs_to_lock` avoids that
by only including specs Pyodide's own lock does not already satisfy.

:func:`mne_pypi_spec` is the same idea for MNE itself: a stable/maint build
can lock the real PyPI release, but a dev build's version has no PyPI release
to lock, so it needs the wheel ``build_lite_wheel.py`` builds instead.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import tomllib
import urllib.request
from pathlib import Path

from jupyterlite_pyodide_kernel.constants import PYODIDE_LOCK_DEFAULT_URL
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PYPROJECT_TOML = Path(__file__).parents[2] / "pyproject.toml"


def jupyterlite_specs():
    """Retrieve the "jupyterlite" dependency group from pyproject.toml."""
    with open(_PYPROJECT_TOML, "rb") as fid:
        return tomllib.load(fid)["dependency-groups"]["jupyterlite"]


def jupyterlite_package_names():
    """Canonical package names in the "jupyterlite" dependency group."""
    return [canonicalize_name(Requirement(spec).name) for spec in jupyterlite_specs()]


def jupyterlite_specs_to_lock():
    """Retrieve specs from "jupyterlite" group Pyodide does not already satisfy."""
    with urllib.request.urlopen(PYODIDE_LOCK_DEFAULT_URL) as fid:
        curated = json.load(fid)["packages"]

    to_lock = []
    for spec in jupyterlite_specs():
        req = Requirement(spec)
        pkg = curated.get(canonicalize_name(req.name))
        if pkg is None or not req.specifier.contains(pkg["version"]):
            to_lock.append(spec)
    return to_lock


def mne_pypi_spec():
    """Return the ``mne==<version>`` spec for a stable/maint build, else None."""
    import mne

    if ".dev" in mne.__version__:
        return None
    return f"mne=={mne.__version__}"
