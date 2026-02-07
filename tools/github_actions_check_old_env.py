"""Check that the old env being used has the expected versions of dependencies."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import importlib
import sys
from importlib import metadata
from pathlib import Path

from packaging.version import Version

project_root = Path(__file__).parent.parent

sys.path.append(project_root / "tools")
from check_pyproject_helpers import (  # noqa: E402
    _get_deps_to_check,
    _get_min_pinned_ver,
)

# Get dependencies to check from pyproject.toml
check_deps = _get_deps_to_check()

# Check that the versions in the env match the minimum versions in pyproject.toml
mod_name_mapping = {"scikit-learn": "sklearn"}
bad_missing = []
bad_version = []
for dep in check_deps:
    mod_name, pyproject_ver = _get_min_pinned_ver(dep)
    mod_import_name = mod_name_mapping.get(mod_name, mod_name)

    # Need to handle logic for checking Python version vs. module versions differently.
    # For Python, the latest micro version for the major.minor release specified will be
    # used. E.g., if we ask for 3.10 when creating the old env, we will get 3.10.19.
    # However, for modules, uv's `lowest-direct` option will resolve to the lowest
    # major.minor.micro version, even if a micro version isn't specified. E.g., if
    # `pyproject.toml` asks for numpy >= 1.26, the lockfile will have 1.26.0.
    if mod_name == "python":
        env_ver = sys.version_info[:3]  # take major, minor, and micro info
        if len(Version(pyproject_ver).release) == 2:  # only major and minor specified
            env_ver = env_ver[:2]  # only compare major and minor info
        env_ver = ".".join(str(x) for x in env_ver)
    else:
        try:
            importlib.import_module(mod_import_name)
        except Exception as exc:
            bad_missing.append(f"{mod_name}: ({type(exc).__name__}: {exc})")
            continue
        # Not all modules have a __version__ attribute, so use importlib.metadata
        # Also requires the true module name, not the import variant (if different)
        env_ver = metadata.version(mod_name)

    if pyproject_ver is None:
        continue  # no min version specified, so no check needed
    if Version(env_ver) != Version(pyproject_ver):
        bad_version.append(
            f"{mod_name}: is {env_ver}; {pyproject_ver} expected from `pyproject.toml`"
        )

if bad_missing:
    bad_missing = "The following module(s) could not be imported:\n" + "\n".join(
        bad_missing
    )
else:
    bad_missing = ""

if bad_version:
    bad_version = (
        "The following module(s) have incorrect versions in the environment:\n"
        + "\n".join(bad_version)
    )
else:
    bad_version = ""

if bad_missing or bad_version:
    raise RuntimeError("\n\n".join([bad_missing, bad_version]))
