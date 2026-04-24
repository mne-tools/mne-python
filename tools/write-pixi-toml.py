import tomlkit
import tomllib

with open("../pyproject.toml", "rb") as fid:
    foo = tomllib.load(fid)

# in keys: build-system, dependency-groups, project, tool

# out keys: workspace, dependencies, pypi-dependencies (may need tasks)
out = dict()
workspace = dict(
    authors=["MNE-Python contributors"],
    channels=["conda-forge"],
    name=foo["project"]["name"],
    platforms=["osx-64"],
    version="0.1.0",
)
dependencies = foo["project"]["dependencies"]
dependencies.extend(foo["dependency-groups"]["test"])
dependencies.extend(foo["dependency-groups"]["test_extra"])
dependencies.extend(foo["project"]["optional-dependencies"]["hdf5"])
dependencies.extend(foo["project"]["optional-dependencies"]["full-no-qt"])
dependencies.extend(foo["project"]["optional-dependencies"]["full-pyside6"])
dependencies = list(
    filter(lambda x: isinstance(x, str) and "mne[" not in x, dependencies)
)
dependencies = sorted(set(dependencies))

out = {"workspace": workspace, "dependencies": dependencies, "pypi-dependencies": []}

with open("pixi-new.toml", "w") as fid:
    tomlkit.dump(out, fid, sort_keys=True)
