# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import tomlkit


def pip_to_pixi(dependencies_list):
    """Convert pip dependency specifier strings to a pixi-style dict."""
    out = dict()
    for dep in dependencies_list:
        if " " in dep:
            name, version = dep.split(" ", maxsplit=1)
        else:
            name, version = dep, "*"
        out[name] = version
    return out


pypi_dependencies = ["pymef"]  # "nest-asyncio2", "pyobjc-framework-Cocoa"]

repo_root = Path(__file__).resolve().parent.parent
pyproject_fpath = repo_root / "pyproject.toml"
pyproject = tomlkit.loads(pyproject_fpath.read_bytes())
outfile = repo_root / "tools" / "ci" / "macos-intel" / "pixi.toml"

# in keys: build-system, dependency-groups, project, tool
# out keys: workspace, dependencies, pypi-dependencies (may need tasks)
out = dict()
workspace = dict(
    authors=["MNE-Python contributors"],
    channels=["conda-forge"],
    name=pyproject["project"]["name"],
    platforms=["osx-64"],
    version="0.1.0",
)
dependencies = pyproject["project"]["dependencies"]
dependencies.extend(pyproject["dependency-groups"]["test"])
dependencies.extend(pyproject["dependency-groups"]["test_extra"])
dependencies.extend(pyproject["project"]["optional-dependencies"]["hdf5"])
dependencies.extend(pyproject["project"]["optional-dependencies"]["full-no-qt"])
dependencies.extend(pyproject["project"]["optional-dependencies"]["full-pyside6"])
dependencies = list(
    filter(lambda x: isinstance(x, str) and "mne[" not in x, dependencies)
)
for ix, dep in enumerate(dependencies):
    if (split_ix := dep.find(";")) >= 0:
        dependencies[ix] = dep[:split_ix].strip()

dependencies = sorted(set(dependencies) - set(pypi_dependencies))

out = {
    "workspace": workspace,
    "dependencies": pip_to_pixi(dependencies),
    "pypi-dependencies": pip_to_pixi(pypi_dependencies),
}
# write the file
outfile.parent.mkdir(parents=True, exist_ok=True)
with open(outfile, "w") as fid:
    tomlkit.dump(out, fid, sort_keys=False)
