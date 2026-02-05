# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import importlib
from importlib import metadata
from pathlib import Path

from packaging.version import Version
from tomlkit.toml_file import TOMLFile

project_root = Path(__file__).parent.parent

# Get dependencies to check from pyproject.toml
pyproject = TOMLFile(project_root / "pyproject.toml")
pyproject_data = pyproject.read()
project_info = pyproject_data["project"]
check_deps = (
    project_info["dependencies"]
    + project_info["optional-dependencies"]["ver-auto-bumped"]
)
n_want_deps = 11  # should be updated when we add more core deps or auto-bumped pins!
assert len(check_deps) == n_want_deps, (
    f"Number of dependencies being checked ({len(check_deps)=}) is not as "
    f"expected {n_want_deps=}"
)

# Check that the versions in the env match the minimum versions in pyproject.toml
package_name_mapping = {"scikit-learn": "sklearn"}
bad_missing = []
bad_version = []
for dep in check_deps:
    components = dep.split(">=")
    if len(components) == 1:
        continue
    package_name = components[0].strip()
    package_import_name = package_name_mapping.get(package_name, package_name)
    pyproject_version = components[1].split(",")[0].strip()

    try:
        mod = importlib.import_module(package_import_name)
    except Exception as exc:
        bad_missing.append(f"{package_name}: ({type(exc).__name__}: {exc})")
        continue
    # Not all packages have a __version__ attribute, so use importlib.metadata instead
    # Also requires the true package name, not the import variant (if different)
    env_ver = metadata.version(package_name)

    if Version(env_ver) != Version(pyproject_version):
        bad_version.append(
            f"{package_name}: is {env_ver}; {pyproject_version} expected from "
            "`pyproject.toml`"
        )

if bad_missing:
    bad_missing = "The following package(s) could not be imported:\n" + "\n".join(
        bad_missing
    )
else:
    bad_missing = ""

if bad_version:
    bad_version = (
        "The following package(s) have incorrect versions in the environment:\n"
        + "\n".join(bad_version)
    )
else:
    bad_version = ""

if bad_missing or bad_version:
    raise RuntimeError("\n\n".join([bad_missing, bad_version]))
