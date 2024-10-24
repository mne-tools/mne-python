#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from pathlib import Path

import tomllib

repo_root = Path(__file__).resolve().parents[2]
with open(repo_root / "pyproject.toml", "rb") as fid:
    pyproj = tomllib.load(fid)

# Get our "full" dependences from `pyproject.toml`, but actually ignore the
# "full" section as it's just "full-noqt" plus PyQt6, and for conda we need PySide
ignore = ("dev", "doc", "test", "test_extra", "full", "full-pyqt6")
deps = set(pyproj["project"]["dependencies"])
for section, section_deps in pyproj["project"]["optional-dependencies"].items():
    if section not in ignore:
        deps |= set(section_deps)
recursive_deps = set(d for d in deps if d.startswith("mne["))
deps -= recursive_deps
deps |= {"pip"}


def remove_spaces(version_spec):
    """Remove spaces in version specs (conda is stricter than pip about this).

    https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/pkg-specs.html#package-match-specifications
    """
    return "".join(version_spec.split())


def split_dep(dep):
    """Separate package name from version spec."""
    pattern = re.compile(r"([^!=<>]+)?([!=<>].*)?")
    groups = list(pattern.match(dep).groups())
    groups[1] = "" if groups[1] is None else remove_spaces(groups[1])
    return tuple(map(str.strip, groups))


# python version
req_python = remove_spaces(pyproj["project"]["requires-python"])

# split package name from version spec
translations = dict(neo="python-neo")
pip_deps = set()
conda_deps = set()
for dep in deps:
    package_name, version_spec = split_dep(dep)
    # handle package name differences
    package_name = translations.get(package_name, package_name)
    # PySide6==6.7.0 only exists on PyPI, not conda-forge, so excluding it in
    # `environment.yaml` breaks the solver
    if package_name == "PySide6":
        version_spec = version_spec.replace("!=6.7.0,", "")
    # rstrip output line in case `version_spec` == ""
    line = f"  - {package_name} {version_spec}".rstrip()
    # use pip for packages needing e.g. `platform_system` or `python_version` triaging
    if ";" in version_spec:
        pip_deps.add(f"    {line}")
    else:
        conda_deps.add(line)

# prepare the pip dependencies section
pip_section = f"""\
  - pip:
{"\n".join(sorted(pip_deps, key=str.casefold))}
"""
pip_section = pip_section if len(pip_deps) else ""
# prepare the env file
env = f"""\
name: mne
channels:
  - conda-forge
dependencies:
  - python {req_python}
{"\n".join(sorted(conda_deps, key=str.casefold))}
{pip_section}"""

(repo_root / "environment.yml").write_text(env)
