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
    get_bad_deps_message,
    get_deps_to_check,
    get_min_pinned_ver,
    raise_bad_deps_messages,
)

# Get dependencies to check from pyproject.toml
check_deps = get_deps_to_check()

# Check that the versions in the env match the minimum versions in pyproject.toml
mod_name_mapping = {"scikit-learn": "sklearn"}
bad_missing = []
bad_version = []
for dep in check_deps:
    mod_name, pyproject_ver = get_min_pinned_ver(dep)
    mod_import_name = mod_name_mapping.get(mod_name, mod_name)

    # Be wary of uv treating lowest Python vs. module versions differently.
    # For Python, the latest micro version for the major.minor release specified will be
    # used. E.g., if we ask for 3.10 when creating the old env, we will get 3.10.19.
    # However, for modules, uv's `lowest-direct` option will resolve to the lowest
    # major.minor.micro version, even if a micro version isn't specified. E.g., if
    # `pyproject.toml` asks for numpy >= 1.26, the lockfile will have 1.26.0.
    # However, the non-lowest micro version of a module may be selected for
    # compatibility reasons. E.g., if `pyproject.toml` asks for pandas >= 2.2 and numpy
    # >= 2.0, pandas 2.2.2 will be placed in the lockfile, as that was the first version
    # to support numpy 2.0. Non-lowest micro versions of modules may also not be used if
    # they were yanked.
    # Therefore, if the micro version of a module isn't specified in `pyproject.toml`,
    # we don't check the micro version of the module in the environment.
    if mod_name == "python":
        env_ver = sys.version_info[:3]  # take major, minor, and micro info
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
    pyproject_ver = Version(pyproject_ver)
    env_ver = Version(env_ver)

    # Discard micro info from env version if it's not specified in pyproject.toml
    if len(pyproject_ver.release) == 2:
        env_ver = Version(f"{env_ver.major}.{env_ver.minor}")

    if env_ver != pyproject_ver:
        bad_version.append(
            f"{mod_name}: is {env_ver}; {pyproject_ver} expected from `pyproject.toml`"
        )

# Format bad messages and raise if there are any bads
bad_missing = get_bad_deps_message(bad_missing, "are missing from the environment")
bad_version = get_bad_deps_message(
    bad_version, "have incorrect versions in the environment"
)
raise_bad_deps_messages([bad_missing, bad_version])
