"""Check that the old env lockfile has the expected versions of dependencies."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
from pathlib import Path

from packaging.specifiers import Specifier
from packaging.version import Version
from tomlkit.toml_file import TOMLFile

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

# Get 'old' lockfile pins for dependencies
lockfile = TOMLFile(project_root / "tools/pylock.ci-old.toml")
lockfile_data = lockfile.read()
python_spec = Specifier(lockfile_data["requires-python"])
assert python_spec.operator == ">=", (
    f"Expected the Python version specifier in `pylock.ci-old.toml` to be a '>=' "
    f"specifier, but found {python_spec.operator}."
)
lockfile_modules = {"python": python_spec.version}
lockfile_modules.update(
    {mod["name"]: mod["version"] for mod in lockfile_data["packages"]}
)

# Check that the versions in the lockfile match the minimum versions in pyproject.toml
# For modules, uv's `lowest-direct` option will resolve to the lowest
# major.minor.micro version, even if a micro version isn't specified. E.g., if
# `pyproject.toml` asks for numpy >= 1.26, the lockfile will have 1.26.0.
# However, the non-lowest micro version of a module may be selected for
# compatibility reasons. E.g., if `pyproject.toml` asks for pandas >= 2.2 and numpy
# >= 2.0, pandas 2.2.2 will be placed in the lockfile, as that was the first version
# to support numpy 2.0. Non-lowest micro versions of modules may also not be used if
# they were yanked.
# Therefore, if the micro version of a module isn't specified in `pyproject.toml`,
# we don't check the micro version of the module in the environment.
mod_name_mapping = {"lazy_loader": "lazy-loader"}
bad_missing = []
bad_version = []
for dep in check_deps:
    mod_name, pyproject_ver = get_min_pinned_ver(dep)
    if pyproject_ver is None:
        continue  # no min version specified, so no check needed
    pyproject_ver = Version(pyproject_ver)
    name = mod_name_mapping.get(mod_name, mod_name)

    if name not in lockfile_modules.keys():
        bad_missing.append(name)
        continue
    lockfile_ver = lockfile_modules[name]
    lockfile_ver = Version(lockfile_ver)

    # Discard micro info from lockfile version if it's not specified in pyproject.toml
    if len(pyproject_ver.release) == 2:
        lockfile_ver = Version(f"{lockfile_ver.major}.{lockfile_ver.minor}")

    if lockfile_ver != pyproject_ver:
        bad_version.append(
            f"lower pin on {name} in `pyproject.toml` is {pyproject_ver}, "
            f"but `pylock.ci-old.toml` has {lockfile_ver}"
        )

# Format bad messages and raise if there are any bads
bad_missing = get_bad_deps_message(
    bad_missing, "are missing from the `pylock.ci-old.toml` lockfile"
)
bad_version = get_bad_deps_message(
    bad_version, "have incorrect versions in the `pylock.ci-old.toml` lockfile"
)
raise_bad_deps_messages([bad_missing, bad_version])
