#!/usr/bin/env python

# Copyright (C) 2011-2020 Alexandre Gramfort
# <alexandre.gramfort@inria.fr>

import os
import os.path as op

from setuptools import setup


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

    # data_dependencies is empty, but let's leave them so that we don't break
    # people's workflows who did `pip install mne[data]`
    install_requires = parse_requirements_file("requirements_base.txt")
    data_requires = []
    hdf5_requires = parse_requirements_file("requirements_hdf5.txt")
    test_requires = parse_requirements_file(
        "requirements_testing.txt"
    ) + parse_requirements_file("requirements_testing_extra.txt")
    setup(
        install_requires=install_requires,
        extras_require={
            "data": data_requires,
            "hdf5": hdf5_requires,
            "test": test_requires,
        },
        packages=package_tree("mne"),
    )
