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
from copy import deepcopy
from pathlib import Path

import requests
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from tomlkit import parse
from tomlkit.items import Comment, Trivia
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
SUPPORT_TIME = datetime.timedelta(days=365 * 2)
CURRENT_DATE = datetime.datetime.now()

project_root = Path(__file__).parent.parent.parent


def get_release_and_drop_dates(package):
    """Get release and drop dates for a given package from pypi.org."""
    releases = {}
    print(f"Querying pypi.org for {package} versions...", end="", flush=True)
    response = requests.get(
        f"https://pypi.org/simple/{package}",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        timeout=10,
    ).json()
    print("OK", flush=True)
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
        cutoff_date = CURRENT_DATE - SUPPORT_TIME
        pre_cutoff = bool(release_date <= cutoff_date)  # was available X time ago
        releases[ver] = {"release_date": release_date, "pre_cutoff": pre_cutoff}
    return releases


def update_specifiers(dependencies, releases):
    """Update dependency version specifiers inplace."""
    for idx, dep in enumerate(dependencies):
        req = Requirement(dep)
        pkg_name = req.name
        pkg_spec = req.specifier
        if pkg_name in releases.keys():  # check if this is a package to update
            # Find package versions matching current specifiers
            package_vers = releases[pkg_name].keys()
            matches = list(pkg_spec.filter(package_vers))  # drop excluded versions
            pre_cutoff_matches = [
                ver for ver in matches if releases[pkg_name][ver]["pre_cutoff"]
            ]
            if len(pre_cutoff_matches) == 0:
                raise RuntimeError(
                    f"{pkg_name} had no versions available {SUPPORT_TIME.days / 365} "
                    "years ago compliant with the following specifier(s): "
                    f"{pkg_spec if pkg_spec else 'None'}",
                )
            post_cutoff_matches = [
                ver for ver in matches if not releases[pkg_name][ver]["pre_cutoff"]
            ]

            # Find latest pre-cutoff version to pin as the minimum version
            min_ver = max(pre_cutoff_matches)
            min_ver, min_ver_release = _find_version_to_pin_and_release(
                min_ver, pkg_spec, pre_cutoff_matches, releases[pkg_name]
            )

            # Find earliest post-cutoff version to pin next
            next_ver = None
            next_ver_release = None
            for ver in post_cutoff_matches:
                if _as_minor_version(ver) > min_ver:  # if a new minor version
                    next_ver, next_ver_release = _find_version_to_pin_and_release(
                        ver, pkg_spec, post_cutoff_matches, releases[pkg_name]
                    )
                    break

            # Update specifiers with new minimum version
            min_ver_spec = SpecifierSet(f">={str(min_ver)}")
            new_spec = [str(min_ver_spec)]
            for spec in str(pkg_spec).split(","):
                spec = spec.strip()
                if spec.startswith(">"):
                    continue  # ignore old min ver
                if spec.startswith("!=") and not min_ver_spec.contains(spec[2:]):
                    continue  # ignore outdated exclusions
                new_spec.append(spec)  # keep max vers and in-date exclusions
            req.specifier = SpecifierSet(",".join(new_spec))

            dependencies._value[idx] = _add_date_comment(
                dependencies._value[idx], min_ver_release, next_ver, next_ver_release
            )
        dependencies[idx] = _prettify_requirement(req)


def _as_minor_version(ver):
    """Convert a version to its major.minor form."""
    return Version(f"{ver.major}.{ver.minor}")


def _find_version_to_pin_and_release(version, specifier, all_versions, release_dates):
    """Find the version to pin based on specifiers and that version's release date.

    Tries to reduce this to an unpatched major.minor form if possible (and use the
    unpatched version's release date). If the unpatched minor form is excluded by the
    specifier, finds the earliest patch version (and its release date) that is not
    excluded.

    E.g., if version=1.2.3 but 1.2.0 is excluded by the specifier, find the earliest
    patched version (e.g., 1.2.1) and pin this. If 1.2.0 is not excluded, we can just
    pin 1.2.

    If the unpatched version is not excluded by the specifier but it has been yanked, we
    don't need to pin the patched version, but we do have to rely on the release date of
    the earliest patched version.
    """
    # Find earliest micro form of this minor version
    version = min(
        ver
        for ver in all_versions
        if _as_minor_version(ver) == _as_minor_version(version)
    )
    # Check unpatched form of this version is not excluded by existing specifiers
    use_patch = not specifier.contains(_as_minor_version(version))
    # Find release date of version to be pinned
    release = release_dates[version]["release_date"]
    # Discard patch info if not needed
    version = _as_minor_version(version) if not use_patch else version

    return version, release


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


def _add_date_comment(dependency, min_ver_release, next_ver, next_ver_release):
    """Add comment for when the min version was released and when it will be changed."""
    comment = f"# released {min_ver_release.strftime('%Y-%m-%d')}"
    if next_ver is not None:
        comment += (
            f", will become {str(next_ver)} on "
            f"{(next_ver_release + SUPPORT_TIME).strftime('%Y-%m-%d')}"
        )
    else:
        comment += ", no newer version available"
    dependency.comment = Comment(
        Trivia(indent="  ", comment_ws="", comment=comment, trail="")
    )
    return dependency


def _find_specifier_order(specifiers):
    """Find ascending order of specifiers according to their version."""
    versions = []
    for spec in specifiers:
        versions.append(Version(re.sub(r"[<>=!~]+", "", str(spec))))  # extract version
    return sorted(range(len(versions)), key=lambda i: versions[i])  # sorted indices


# Find release and drop dates for desired packages
package_releases = {
    package: get_release_and_drop_dates(package) for package in SORT_PACKAGES
}

# Get dependencies from pyproject.toml
pyproject = TOMLFile(project_root / "pyproject.toml")
pyproject_data = pyproject.read()
project_info = pyproject_data["project"]
core_dependencies = project_info["dependencies"]
opt_dependencies = project_info["optional-dependencies"]

# Update version specifiers
changed = []
old_deps = deepcopy(core_dependencies)
update_specifiers(core_dependencies, package_releases)
changed.extend(
    [
        f"Core dependency ``{new}``"
        for new, old in zip(core_dependencies, old_deps)
        if new != old
    ]
)
for key in opt_dependencies:
    old_deps = deepcopy(opt_dependencies[key])
    update_specifiers(opt_dependencies[key], package_releases)
    changed.extend(
        [
            f"Optional dependency ``{new}``"
            for new, old in zip(opt_dependencies[key], old_deps)
            if new != old
        ]
    )

# Need to write a changelog entry if versions were updated
if changed:
    changelog_text = "Updated minimum for:\n\n"
    changelog_text += "\n".join(f"- {change}" for change in changed)
    print(changelog_text, flush=True)
    # no reason to print this but it should go in the changelog
    changelog_text += (
        "\n\nChanged implemented via CI action created by `Thomas Binns`_.\n"
    )
    changelog_path = project_root / "doc" / "changes" / "dev" / "dependency.rst"
    changelog_path.write_text(changelog_text, encoding="utf-8")
else:
    print("No dependency versions needed updating.", flush=True)

# Save updated pyproject.toml (replace ugly \" with ' first)
pyproject_data = parse(pyproject_data.as_string().replace('\\"', "'"))
pyproject.write(pyproject_data)
