# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import subprocess
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

import tomllib

parser = ArgumentParser(description="Generate codemeta.json and CITATION.cff")
parser.add_argument("release_version", type=str)
release_version = parser.parse_args().release_version

out_dir = Path(__file__).parents[1]

# NOTE: ../codemeta.json and ../citation.cff should not be continuously
#       updated. Run this script only at release time.

package_name = "MNE-Python"
release_date = str(date.today())
commit = subprocess.run(
    ["git", "log", "-1", "--pretty=%H"], capture_output=True, text=True
).stdout.strip()

# KEYWORDS
keywords = (
    "MEG",
    "magnetoencephalography",
    "EEG",
    "electroencephalography",
    "fNIRS",
    "functional near-infrared spectroscopy",
    "iEEG",
    "intracranial EEG",
    "eCoG",
    "electrocorticography",
    "DBS",
    "deep brain stimulation",
)

# add to these as necessary
compound_surnames = (
    "García Alanis",
    "van Vliet",
    "De Santis",
    "Dupré la Tour",
    "de la Torre",
    "de Jong",
    "de Montalivet",
    "van den Bosch",
    "Van den Bossche",
    "Van Der Donckt",
    "van der Meer",
    "van Harmelen",
    "Visconti di Oleggio Castello",
    "van Es",
)


def parse_name(name):
    """Split name blobs from `git shortlog -nse` into first/last/email."""
    # remove commit count
    _, name_and_email = name.strip().split("\t")
    name, email = name_and_email.split(" <")
    email = email.strip(">")
    email = "" if "noreply" in email else email  # ignore "noreply" emails
    name = " ".join(name.split("."))  # remove periods from initials
    # handle compound surnames
    for compound_surname in compound_surnames:
        if name.endswith(compound_surname):
            ix = name.index(compound_surname)
            first = name[:ix].strip()
            last = compound_surname
            return (first, last, email)
    # handle non-compound surnames
    name_elements = name.split()
    if len(name_elements) == 1:  # mononyms / usernames
        first = ""
        last = name
    else:
        first = " ".join(name_elements[:-1])
        last = name_elements[-1]
    return (first, last, email)


# MAKE SURE THE RELEASE STRING IS PROPERLY FORMATTED
try:
    split_version = list(map(int, release_version.split(".")))
except ValueError:
    raise
msg = (
    "First argument must be the release version X.Y.Z (all integers), "
    f"got {release_version}"
)
assert len(split_version) == 3, msg


# RUN GIT SHORTLOG TO GET ALL AUTHORS, SORTED BY NUMBER OF COMMITS
args = ["git", "shortlog", "-nse"]
result = subprocess.run(args, capture_output=True, text=True)
lines = result.stdout.strip().split("\n")
all_names = [parse_name(line) for line in lines if "[bot]" not in line]


# CONSTRUCT JSON AUTHORS LIST
json_authors = [
    f"""{{
           "@type":"Person",
           "email":"{email}",
           "givenName":"{first}",
           "familyName": "{last}"
        }}"""
    for (first, last, email) in all_names
]


# GET OUR DEPENDENCY VERSIONS
pyproject = tomllib.loads(
    (Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8")
)
dependencies = [f"python{pyproject['project']['requires-python']}"]
dependencies.extend(pyproject["project"]["dependencies"])

# these must be done outside the boilerplate (no \n allowed in f-strings):
json_authors = ",\n        ".join(json_authors)
dependencies = '",\n        "'.join(dependencies)
json_keywords = '",\n        "'.join(keywords)


# ASSEMBLE COMPLETE JSON
codemeta_boilerplate = f"""{{
    "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
    "@type": "SoftwareSourceCode",
    "license": "https://spdx.org/licenses/BSD-3-Clause",
    "codeRepository": "git+https://github.com/mne-tools/mne-python.git",
    "dateCreated": "2010-12-26",
    "datePublished": "2014-08-04",
    "dateModified": "{release_date}",
    "downloadUrl": "https://github.com/mne-tools/mne-python/archive/v{release_version}.zip",
    "issueTracker": "https://github.com/mne-tools/mne-python/issues",
    "name": "{package_name}",
    "version": "{release_version}",
    "description": "{package_name} is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data. It provides methods for data input/output, preprocessing, visualization, source estimation, time-frequency analysis, connectivity analysis, machine learning, and statistics.",
    "applicationCategory": "Neuroscience",
    "developmentStatus": "active",
    "referencePublication": "https://doi.org/10.3389/fnins.2013.00267",
    "keywords": [
        "{json_keywords}"
    ],
    "programmingLanguage": [
        "Python"
    ],
    "operatingSystem": [
        "Linux",
        "Windows",
        "macOS"
    ],
    "softwareRequirements": [
        "{dependencies}"
    ],
    "author": [
        {json_authors}
    ]
}}
"""  # noqa E501


# WRITE TO FILE
with open(out_dir / "codemeta.json", "w") as codemeta_file:
    codemeta_file.write(codemeta_boilerplate)


# # # # # # # # # # # # # # #
# GENERATE CITATION.CFF TOO #
# # # # # # # # # # # # # # #
message = (
    "If you use this software, please cite both the software itself, "
    "and the paper listed in the preferred-citation field."
)

# in CFF, multi-word keywords need to be wrapped in quotes
cff_keywords = (f'"{kw}"' if " " in kw else kw for kw in keywords)
# make into a bulleted list
cff_keywords = "\n".join(f"  - {kw}" for kw in cff_keywords)

# TODO: someday would be nice to include ORCiD identifiers too
cff_authors = [
    f"  - family-names: {last}\n    given-names: {first}"
    if first
    else f"  - name: {last}"
    for (first, last, _) in all_names
]
cff_authors = "\n".join(cff_authors)

# this ↓↓↓ is the meta-DOI that always resolves to the latest release
zenodo_doi = "10.5281/zenodo.592483"

# ASSEMBLE THE CFF STRING
cff_boilerplate = f"""\
cff-version: 1.2.0
title: "{package_name}"
message: "{message}"
version: {release_version}
date-released: "{release_date}"
commit: {commit}
doi: {zenodo_doi}
keywords:
{cff_keywords}
authors:
{cff_authors}
preferred-citation:
  title: "MEG and EEG Data Analysis with MNE-Python"
  journal: "Frontiers in Neuroscience"
  type: article
  year: 2013
  volume: 7
  issue: 267
  start: 1
  end: 13
  doi: 10.3389/fnins.2013.00267
  authors:
    - family-names: Gramfort
      given-names: Alexandre
    - family-names: Luessi
      given-names: Martin
    - family-names: Larson
      given-names: Eric
    - family-names: Engemann
      given-names: Denis A.
    - family-names: Strohmeier
      given-names: Daniel
    - family-names: Brodbeck
      given-names: Christian
    - family-names: Goj
      given-names: Roman
    - family-names: Jas
      given-names: Mainak
    - family-names: Brooks
      given-names: Teon
    - family-names: Parkkonen
      given-names: Lauri
    - family-names: Hämäläinen
      given-names: Matti S.
"""

# WRITE TO FILE
with open(out_dir / "CITATION.cff", "w") as cff_file:
    cff_file.write(cff_boilerplate)
