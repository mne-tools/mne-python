import os
import subprocess

# UPDATE THESE AS NEEDED WITH EACH RELEASE
release_date = '2020-12-17'
release_version = '0.22'
dependencies = ['Python 3.6', 'NumPy 1.15.4', 'SciPy 1.1']
# add to these as necessary
compound_surnames = (
    'García Alanis',
    'van Vliet',
    'De Santis',
    'Dupré la Tour',
    'de la Torre',
    'van den Bosch',
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
    name = ' '.join(name.split('.'))             # remove periods
    # handle compound surnames
    for compound_surname in compound_surnames:
        if name.endswith(compound_surname):
            ix = name.index(compound_surname)
            first = name[:ix].strip()
            last = compound_surname
            return (first, last, email)
    # handle non-compound surnames
    names = name.split()
    if len(names) == 1:  # mononyms / usernames
        first = ''
        last = name
    else:
        first = ' '.join(names[:-1])
        last = names[-1]
    return (first, last, email)


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
    "dateModified": "{release_date}",
    "downloadUrl": "https://github.com/mne-tools/mne-python/archive/v0.22.0.zip",
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
        "eCoG",
        "sEEG"
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
