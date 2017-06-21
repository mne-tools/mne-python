#! /usr/bin/env python
#

# Copyright (C) 2011-2014 Alexandre Gramfort
# <alexandre.gramfort@telecom-paristech.fr>

import os
from os import path as op

from setuptools import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('mne', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """MNE python project for MEG and EEG data analysis."""

DISTNAME = 'mne'
DESCRIPTION = descr
MAINTAINER = 'Alexandre Gramfort'
MAINTAINER_EMAIL = 'alexandre.gramfort@telecom-paristech.fr'
URL = 'http://martinos.org/mne'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-python'
VERSION = version


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=package_tree('mne'),
          package_data={'mne': [op.join('data', '*.sel'),
                                op.join('data', 'icos.fif.gz'),
                                op.join('data', 'coil_def*.dat'),
                                op.join('data', 'helmets', '*.fif.gz'),
                                op.join('data', 'FreeSurferColorLUT.txt'),
                                op.join('data', 'image', '*gif'),
                                op.join('data', 'image', '*lout'),
                                op.join('data', 'fsaverage', '*.fif'),
                                op.join('channels', 'data', 'layouts', '*.lout'),
                                op.join('channels', 'data', 'layouts', '*.lay'),
                                op.join('channels', 'data', 'montages', '*.sfp'),
                                op.join('channels', 'data', 'montages', '*.txt'),
                                op.join('channels', 'data', 'montages', '*.elc'),
                                op.join('channels', 'data', 'neighbors', '*.mat'),
                                op.join('gui', 'help', '*.json'),
                                op.join('html', '*.js'),
                                op.join('html', '*.css'),
                                op.join('io', 'artemis123', 'resources', '*.csv')
                                ]},
          scripts=['bin/mne'])
