"""Update dependency version specifiers to comply with SPEC0."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections
import datetime
import re
import warnings

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from tomlkit.toml_file import TOMLFile

SORT_PACKAGES = ["matplotlib", "numpy", "pandas", "pyvista", "pyvistaqt", "scipy"]
PLUS_24_MONTHS = datetime.timedelta(days=int(365 * 2))

# Release data
CURRENT_DATE = datetime.datetime.now()


def get_release_and_drop_dates(package, support_time=PLUS_24_MONTHS):
    """Get release and drop dates for a given package from pypi.org."""
    releases = {}
    print(f"Querying pypi.org for {package} versions...", end="", flush=True)
    response = requests.get(
        f"https://pypi.org/simple/{package}",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    ).json()
    print("OK")
    file_date = collections.defaultdict(list)
    for f in response["files"]:
        if f["filename"].endswith(".tar.gz") or f["filename"].endswith(".zip"):
            continue
        if f["yanked"]:
            continue
        ver = f["filename"].split("-")[1]
        try:
            version = Version(ver)
        except InvalidVersion:
            continue
        if version.is_prerelease:
            continue
        release_date = datetime.datetime.fromisoformat(f["upload-time"]).replace(
            tzinfo=None
        )
        if not release_date:
            continue
        file_date[version].append(release_date)
    release_date = {v: min(file_date[v]) for v in file_date}
    for ver, release_date in sorted(release_date.items()):
        drop_date = release_date + support_time
        if drop_date > CURRENT_DATE:
            releases[ver] = {
                "release_date": release_date,
                "drop_date": drop_date,
            }
    return releases


def update_specifiers(dependencies, releases):
    """Update dependency version specifiers."""
    for idx, dep in enumerate(dependencies):
        req = Requirement(dep)
        if req.name in releases.keys():  # check if this is a package to update
            package_vers = releases[req.name].keys()
            spec_matches = list(req.specifier.filter(package_vers))
            if len(spec_matches) == 0:
                warnings.warn(
                    f"Dependency has no valid versions.\n"
                    f"  name: {req.name}\n"
                    f"  specifier(s): {req.specifier if req.specifier else 'None'}",
                    RuntimeWarning,
                )
                continue
            min_ver = SpecifierSet(f">={str(min(spec_matches))}")
            new_spec = [str(min_ver)]
            for spec in str(req.specifier).split(","):
                spec = spec.strip()
                if spec.startswith(">"):
                    continue  # ignore old min ver
                if spec.startswith("!=") and not min_ver.contains(spec[2:]):
                    continue  # ignore outdated exclusions
                new_spec.append(spec)  # max vers and in-date exclusions
            req.specifier = SpecifierSet(",".join(new_spec))
        dependencies[idx] = _prettify_requirement(req)
    return dependencies


def _prettify_requirement(req):
    """Add spacing to make a requirement specifier prettier."""
    specifiers = ""
    for spec in req.specifier:
        spec = str(spec)
        split = re.search(r"[<>=]\d", spec).span()[1] - 1  # find end of operator
        specifiers += f" {spec[:split]} {spec[split:]},"  # pad operator with spaces
    specifiers = specifiers.rstrip(",")  # remove trailing comma
    return req.name + specifiers


package_releases = {
    package: get_release_and_drop_dates(package) for package in SORT_PACKAGES
}

pyproject_data = TOMLFile("pyproject.toml").read()
project_info = pyproject_data.get("project")
core_dependencies = project_info["dependencies"]
opt_dependencies = project_info.get("optional-dependencies", {})

core_dependencies = update_specifiers(core_dependencies, package_releases)
for key in opt_dependencies:
    opt_dependencies[key] = update_specifiers(opt_dependencies[key], package_releases)

pyproject_data["project"]["dependencies"] = core_dependencies
if opt_dependencies:
    pyproject_data["project"]["optional-dependencies"] = opt_dependencies

TOMLFile("pyproject.toml").write(pyproject_data)
