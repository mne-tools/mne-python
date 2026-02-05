# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

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

# Get 'old' lockfile pins for dependencies
lockfile = TOMLFile(project_root / "tools/pylock.ci-old.toml")
lockfile_data = lockfile.read()
lockfile_packages = {pkg["name"]: pkg["version"] for pkg in lockfile_data["packages"]}

# Check that the versions in the lockfile match the minimum versions in pyproject.toml
package_name_mapping = {"lazy_loader": "lazy-loader"}
bad_missing = []
bad_version = []
for dep in check_deps:
    components = dep.split(">=")
    if len(components) == 1:
        continue
    package = components[0].strip()
    package = package_name_mapping.get(package, package)
    pyproject_version = components[1].split(",")[0].strip()

    if package not in lockfile_packages.keys():
        bad_missing.append(package)
        continue
    lockfile_version = lockfile_packages[package]

    if Version(lockfile_version) != Version(pyproject_version):
        bad_version.append(
            f"`pyproject.toml` expects {package} == {pyproject_version}, but "
            f"`pylock.ci-old.toml` has {lockfile_version}"
        )

if bad_missing:
    bad_missing = (
        "The following package(s) are missing from the `pylock.ci-old.toml` lockfile: "
        + ", ".join(bad_missing)
    )
else:
    bad_missing = ""

if bad_version:
    bad_version = (
        "The following package(s) have incorrect versions in the `pylock.ci-old.toml` "
        "lockfile:\n" + "\n".join(bad_version)
    )
else:
    bad_version = ""

if bad_missing or bad_version:
    raise RuntimeError("\n\n".join([bad_missing, bad_version]))
