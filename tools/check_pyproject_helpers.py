"""Checks for handling version pins for dependencies in `pyproject.toml`."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from packaging.requirements import Requirement
from tomlkit.toml_file import TOMLFile

project_root = Path(__file__).parent.parent


def get_deps_to_check():
    """Get the dependencies whose versions should be checked from `pyproject.toml`."""
    pyproject = TOMLFile(project_root / "pyproject.toml")
    pyproject_data = pyproject.read()
    check_deps = (
        [f"python {pyproject_data['project']['requires-python']}"]
        + pyproject_data["project"]["dependencies"]
        + pyproject_data["dependency-groups"]["lockfile_extras"]
    )
    n_want_deps = 12  # update when we add more core deps or auto-bumped pins!
    assert len(check_deps) == n_want_deps, (
        f"Number of dependencies being checked ({len(check_deps)=}) is not as "
        f"expected {n_want_deps=}"
    )

    return check_deps


def get_min_pinned_ver(req):
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


def get_bad_deps_message(bads, bads_reason):
    """Format a message about bad deps (e.g., missing, wrong version), if any."""
    if len(bads) == 0:
        return ""
    return f"The following module(s) {bads_reason}:\n" + "\n".join(bads)


def raise_bad_deps_messages(bad_messages):
    """Raise a RuntimeError if there are any bad messages to report."""
    bad_messages = [message for message in bad_messages if message != ""]
    if len(bad_messages) > 0:
        raise RuntimeError("\n\n".join(bad_messages))
