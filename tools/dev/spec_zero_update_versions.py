"""Update version specifiers of certain dependencies to comply with SPEC0-like rules.

SPEC0 (https://scientific-python.org/specs/spec-0000/) recommends that support for
Python versions are dropped 3 years after initial release, and for core package
dependencies 2 years after initial release.

MNE-Python follows a SPEC0-like policy that reduces maintenance burden whilst
accommodating users in minimum version support similarly to when before this policy was
adopted.

MNE-Python's policy differs from SPEC0 in the following ways:
- Python versions are supported for at least 3 years after release, but possibly longer
  at the discretion of the MNE-Python maintainers based on, e.g., maintainability,
  features.
- Not all core dependencies have minimum versions pinned, and some optional dependencies
  have minimum versions pinned. Only those dependencies whose older versions require
  considerable work to maintain compatibility with (e.g., due to API changes) have
  minimum versions pinned.
- Micro/patch versions are not pinned as minimum versions (unless there is an issue with
  a specific patch), as these should not introduce breaking changes.
- Minimum versions for dependencies are set to the latest minor release that was
  available 2 years prior. The rationale behind this is discussed here:
  https://github.com/mne-tools/mne-python/pull/13451#discussion_r2445337934

For example, in October 2025:
- The latest version of NumPy available 2 years prior was 1.26.1 (released October
  2023), making the latest minor release 1.26, which would be pinned. Support for 1.26
  would be dropped in June 2026 in favour of 2.0, which was released in June 2024.
- The latest version of SciPy available 2 years prior was 1.11.3 (release September
  2023), making the latest minor release 1.11, which would be pinned. Support for 1.11
  would be dropped in January 2026 in favour of 1.12, which was released in January
  2024.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import collections
import datetime
import re

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from tomlkit import parse
from tomlkit.toml_file import TOMLFile

SORT_PACKAGES = [
    "matplotlib",
    "numpy",
    "pandas",
    "pyvista",
    "pyvistaqt",
    "scikit-learn",
    "scipy",
]
PLUS_24_MONTHS = datetime.timedelta(days=365 * 2)
CURRENT_DATE = datetime.datetime.now()


def get_release_and_drop_dates(package, support_time):
    """Get release and drop dates for a given package from pypi.org."""
    releases = {}
    print(f"Querying pypi.org for {package} versions...", end="", flush=True)
    response = requests.get(
        f"https://pypi.org/simple/{package}",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        timeout=10,
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
            releases[ver] = {"release_date": release_date, "drop_date": drop_date}
    return releases


def update_specifiers(dependencies, releases):
    """Update dependency version specifiers."""
    for idx, dep in enumerate(dependencies):
        req = Requirement(dep)
        if req.name in releases.keys():  # check if this is a package to update
            package_vers = releases[req.name].keys()
            spec_matches = list(req.specifier.filter(package_vers))
            if len(spec_matches) == 0:
                raise RuntimeError(
                    f"Dependency has no valid versions.\n"
                    f"  name: {req.name}\n"
                    f"  specifier(s): {req.specifier if req.specifier else 'None'}",
                )
            min_ver = min(spec_matches)  # find earliest valid version
            min_ver = Version(f"{min_ver.major}.{min_ver.minor}")  # ignore patches
            min_ver = SpecifierSet(f">={str(min_ver)}")
            new_spec = [str(min_ver)]
            for spec in str(req.specifier).split(","):
                spec = spec.strip()
                if spec.startswith(">"):
                    continue  # ignore old min ver
                if spec.startswith("!=") and not min_ver.contains(spec[2:]):
                    continue  # ignore outdated exclusions
                new_spec.append(spec)  # keep max vers and in-date exclusions
            req.specifier = SpecifierSet(",".join(new_spec))
        dependencies[idx] = _prettify_requirement(req)
    return dependencies


def _prettify_requirement(req):
    """Add spacing to make a requirement specifier prettier."""
    specifiers = []
    spec_order = _find_specifier_order(req.specifier)
    for spec in req.specifier:
        spec = str(spec)
        split = re.search(r"[<>=]\d", spec).span()[1] - 1  # find end of operator
        specifiers.append(f" {spec[:split]} {spec[split:]},")  # pad operator w/ spaces
    specifiers = [specifiers[i] for i in spec_order]  # order by ascending version
    specifiers = "".join(specifiers)
    specifiers = specifiers.rstrip(",")  # remove trailing comma
    req.specifier = SpecifierSet()  # remove ugly specifiers (from str repr)
    # Add pretty specifiers to name alongside trailing info (extras, markers, url)
    return req.name + specifiers + str(req)[len(req.name) :]


def _find_specifier_order(specifiers):
    """Find ascending order of specifiers according to their version."""
    versions = []
    for spec in specifiers:
        versions.append(Version(re.sub(r"[<>=!~]+", "", str(spec))))  # extract version
    return sorted(range(len(versions)), key=lambda i: versions[i])  # sorted indices


# Find release and drop dates for desired packages
package_releases = {
    package: get_release_and_drop_dates(package, support_time=PLUS_24_MONTHS)
    for package in SORT_PACKAGES
}

# Get dependencies from pyproject.toml
pyproject = TOMLFile("pyproject.toml")
pyproject_data = pyproject.read()
project_info = pyproject_data.get("project")
core_dependencies = project_info["dependencies"]
opt_dependencies = project_info.get("optional-dependencies", {})

# Update version specifiers
core_dependencies = update_specifiers(core_dependencies, package_releases)
for key in opt_dependencies:
    opt_dependencies[key] = update_specifiers(opt_dependencies[key], package_releases)
pyproject_data["project"]["dependencies"] = core_dependencies
if opt_dependencies:
    pyproject_data["project"]["optional-dependencies"] = opt_dependencies

# Save updated pyproject.toml (replace ugly \" with ' first)
pyproject_data = parse(pyproject_data.as_string().replace('\\"', "'"))
pyproject.write(pyproject_data)
