#!/usr/bin/env python

# Copyright (C) 2011-2020 Alexandre Gramfort
# <alexandre.gramfort@inria.fr>

import os
import os.path as op

from setuptools import setup


if "SETUPTOOLS_SCM_PRETEND_VERSION" not in os.environ:
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = "1.4.0"


def parse_requirements_file(fname):
    requirements = list()
    with open(fname, "r") as fid:
        for line in fid:
            req = line.strip()
            if req.startswith("#"):
                continue
            # strip end-of-line comments
            req = req.split("#", maxsplit=1)[0].strip()
            requirements.append(req)
    return requirements


DISTNAME = "mne"
DESCRIPTION = "MNE-Python project for MEG and EEG data analysis."
MAINTAINER = "Alexandre Gramfort"
MAINTAINER_EMAIL = "alexandre.gramfort@inria.fr"
URL = "https://mne.tools/dev/"
LICENSE = "BSD-3-Clause"
DOWNLOAD_URL = "http://github.com/mne-tools/mne-python"


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = op.dirname(__file__)
    subdirs = [
        op.relpath(i[0], path).replace(op.sep, ".")
        for i in os.walk(op.join(path, pkgroot))
        if "__init__.py" in i[2]
    ]
    return sorted(subdirs)


if __name__ == "__main__":
    if op.exists("MANIFEST"):
        os.remove("MANIFEST")

    with open("README.rst", "r") as fid:
        long_description = fid.read()

    # data_dependencies is empty, but let's leave them so that we don't break
    # people's workflows who did `pip install mne[data]`
    install_requires = parse_requirements_file("requirements_base.txt")
    data_requires = []
    hdf5_requires = parse_requirements_file("requirements_hdf5.txt")
    test_requires = parse_requirements_file(
        "requirements_testing.txt"
    ) + parse_requirements_file("requirements_testing_extra.txt")
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        long_description=long_description,
        long_description_content_type="text/x-rst",
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
        ],
        keywords="neuroscience neuroimaging MEG EEG ECoG fNIRS brain",
        project_urls={
            "Homepage": "https://mne.tools/",
            "Download": "https://pypi.org/project/scikit-learn/#files",
            "Bug Tracker": "https://github.com/mne-tools/mne-python/issues/",
            "Documentation": "https://mne.tools/",
            "Forum": "https://mne.discourse.group/",
            "Source Code": "https://github.com/mne-tools/mne-python/",
        },
        platforms="any",
        python_requires=">=3.8",
        install_requires=install_requires,
        setup_requires=["setuptools>=45", "setuptools_scm>=6.2"],
        use_scm_version={
            "write_to": "mne/_version.py",
            "version_scheme": "release-branch-semver",
        },
        extras_require={
            "data": data_requires,
            "hdf5": hdf5_requires,
            "test": test_requires,
        },
        packages=package_tree("mne"),
        package_data={
            "mne": [
                op.join("data", "eegbci_checksums.txt"),
                op.join("data", "*.sel"),
                op.join("data", "icos.fif.gz"),
                op.join("data", "coil_def*.dat"),
                op.join("data", "helmets", "*.fif.gz"),
                op.join("data", "FreeSurferColorLUT.txt"),
                op.join("data", "image", "*gif"),
                op.join("data", "image", "*lout"),
                op.join("data", "fsaverage", "*.fif"),
                op.join("channels", "data", "layouts", "*.lout"),
                op.join("channels", "data", "layouts", "*.lay"),
                op.join("channels", "data", "montages", "*.sfp"),
                op.join("channels", "data", "montages", "*.txt"),
                op.join("channels", "data", "montages", "*.elc"),
                op.join("channels", "data", "neighbors", "*.mat"),
                op.join("datasets", "sleep_physionet", "SHA1SUMS"),
                op.join("datasets", "_fsaverage", "*.txt"),
                op.join("datasets", "_infant", "*.txt"),
                op.join("datasets", "_phantom", "*.txt"),
                op.join("html", "*.js"),
                op.join("html", "*.css"),
                op.join("html_templates", "repr", "*.jinja"),
                op.join("html_templates", "report", "*.jinja"),
                op.join("icons", "*.svg"),
                op.join("icons", "*.png"),
                op.join("io", "artemis123", "resources", "*.csv"),
                op.join("io", "edf", "gdf_encodes.txt"),
            ]
        },
        entry_points={
            "console_scripts": [
                "mne = mne.commands.utils:main",
            ]
        },
    )
