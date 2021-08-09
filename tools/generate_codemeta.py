import os
import subprocess
from datetime import date
from mne import __version__ as release_version

# NOTE: ../codemeta.json should not be continuously updated. Run this script
#       only at release time.

# add to these as necessary
compound_surnames = (
    'García Alanis',
    'van Vliet',
    'De Santis',
    'Dupré la Tour',
    'de la Torre',
    'van den Bosch',
    'Van den Bossche',
    'Van Der Donckt',
    'van der Meer',
    'van Harmelen',
    'Visconti di Oleggio Castello'
)


def parse_name(name):
    """Split name blobs from `git shortlog -nse` into first/last/email."""
    # remove commit count
    _, name_and_email = name.strip().split('\t')
    name, email = name_and_email.split(' <')
    email = email.strip('>')
    email = '' if 'noreply' in email else email  # ignore "noreply" emails
    name = ' '.join(name.split('.'))             # remove periods from initials
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
        first = ''
        last = name
    else:
        first = ' '.join(name_elements[:-1])
        last = name_elements[-1]
    return (first, last, email)


# MAKE SURE THE RELEASE STRING IS PROPERLY FORMATTED
try:
    split_version = list(map(int, release_version.split('.')))
except ValueError:
    raise
msg = f'version string must be X.Y.Z (all integers), got {release_version}'
assert len(split_version) == 3, msg


# RUN GIT SHORTLOG TO GET ALL AUTHORS, SORTED BY NUMBER OF COMMITS
args = ['git', 'shortlog', '-nse']
result = subprocess.run(args, capture_output=True, text=True)
lines = result.stdout.strip().split('\n')
all_names = [parse_name(line) for line in lines]


# CONSTRUCT JSON AUTHORS LIST
authors = [f'''{{
           "@type":"Person",
           "email":"{email}",
           "givenName":"{first}",
           "familyName": "{last}"
        }}''' for (first, last, email) in all_names]


# GET OUR DEPENDENCIES
with open(os.path.join('..', 'setup.py'), 'r') as fid:
    for line in fid:
        if line.strip().startswith('python_requires='):
            version = line.strip().split('=', maxsplit=1)[1].strip("'\",")
            dependencies = [f'python{version}']
            break
hard_dependencies = ('numpy', 'scipy')
with open(os.path.join('..', 'requirements.txt'), 'r') as fid:
    for line in fid:
        req = line.strip()
        for hard_dep in hard_dependencies:
            if req.startswith(hard_dep):
                dependencies.append(req)


# these must be done outside the boilerplate (no \n allowed in f-strings):
authors = ',\n        '.join(authors)
dependencies = '",\n        "'.join(dependencies)


# ASSEMBLE COMPLETE JSON
codemeta_boilerplate = f'''{{
    "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
    "@type": "SoftwareSourceCode",
    "license": "https://spdx.org/licenses/BSD-3-Clause",
    "codeRepository": "git+https://github.com/mne-tools/mne-python.git",
    "dateCreated": "2010-12-26",
    "datePublished": "2014-08-04",
    "dateModified": "{str(date.today())}",
    "downloadUrl": "https://github.com/mne-tools/mne-python/archive/v{release_version}.zip",
    "issueTracker": "https://github.com/mne-tools/mne-python/issues",
    "name": "MNE-Python",
    "version": "{release_version}",
    "description": "MNE-Python is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data. It provides methods for data input/output, preprocessing, visualization, source estimation, time-frequency analysis, connectivity analysis, machine learning, and statistics.",
    "applicationCategory": "Neuroscience",
    "developmentStatus": "active",
    "referencePublication": "https://doi.org/10.3389/fnins.2013.00267",
    "keywords": [
        "MEG",
        "EEG",
        "fNIRS",
        "ECoG",
        "sEEG",
        "DBS"
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
        {authors}
    ]
}}
'''  # noqa E501


# WRITE TO FILE
with open(os.path.join('..', 'codemeta.json'), 'w') as codemeta_file:
    codemeta_file.write(codemeta_boilerplate)
