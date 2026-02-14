"""Checks for handling version pins for dependencies in `pyproject.toml`."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from packaging.requirements import Requirement
from tomlkit.toml_file import TOMLFile

project_root = Path(__file__).parent.parent


def _get_deps_to_check():
    """Get the dependencies whose versions should be checked from `pyproject.toml`."""
    pyproject = TOMLFile(project_root / "pyproject.toml")
    pyproject_data = pyproject.read()
    project_info = pyproject_data["project"]
    check_deps = (
        [f"python {project_info['requires-python']}"]
        + project_info["dependencies"]
        + project_info["optional-dependencies"]["ver-auto-bumped"]
    )
    n_want_deps = 12  # update when we add more core deps or auto-bumped pins!
    assert len(check_deps) == n_want_deps, (
        f"Number of dependencies being checked ({len(check_deps)=}) is not as "
        f"expected {n_want_deps=}"
    )

    return check_deps


def _get_min_pinned_ver(req):
    """Get the min pinned version from a dep specification in `pyproject.toml`."""
    req = Requirement(req)
    name = req.name
    spec = req.specifier
    if len(spec) == 0:
        return name, None  # no min version specified
    ge_specs = [this_spec for this_spec in spec if this_spec.operator == ">="]
    assert len(ge_specs) == 1, (
        f"Expected exactly 1 '>=' specifier in `pyproject.toml` for module {name} with "
        f"version specifications, but found {len(ge_specs)}"
        f"{': ' + ', '.join([str(ge_spec) for ge_spec in ge_specs]) if len(ge_specs) > 0 else ''}."  # noqa: E501
    )  # can't use \ to break f-string statements until python 3.12

    return name, ge_specs[0].version
