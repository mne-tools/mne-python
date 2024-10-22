#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import tomllib
import yaml

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

# python version
req_python = "".join(pyproj["project"]["requires-python"].split())
py_spec = f"python {req_python}"

# load extra packages that *can't* be in `pyproject.toml`
with open(".conda-extra-dependencies.yml") as fid:
    deps |= set(yaml.safe_load(fid))

# handle package name differences, and split package name from version spec
translations = dict(neo="python-neo")
pip_deps = set()
conda_deps = set()
for dep in deps:
    package_name, version_spec, *_ = [*dep.split(maxsplit=1), ""]
    package_name = translations.get(package_name, package_name)
    # collapse spaces in version spec (cf. https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/pkg-specs.html#package-match-specifications)  # noqa E501
    version_spec = "".join(version_spec.split())
    line = f"  - {package_name} {version_spec}".rstrip()
    # use pip for anything that needs e.g. `platform_system=='Darwin'` triaging
    if "platform_system" in version_spec:
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
  - {py_spec}
{"\n".join(sorted(conda_deps, key=str.casefold))}
{pip_section}"""

with open(repo_root / "environment.yml", "w") as fid:
    fid.write(env)
