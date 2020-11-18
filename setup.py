#!/usr/bin/env python

# Copyright (C) 2011-2020 Alexandre Gramfort
# <alexandre.gramfort@inria.fr>

import os
import os.path as op

from setuptools import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('mne', '_version.py'), 'r') as fid:
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
MAINTAINER_EMAIL = 'alexandre.gramfort@inria.fr'
URL = 'https://mne.tools/dev/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-python'
VERSION = version


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


if __name__ == "__main__":
    if op.exists('MANIFEST'):
        os.remove('MANIFEST')

    with open('README.rst', 'r') as fid:
        long_description = fid.read()

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=long_description,
          long_description_content_type='text/x-rst',
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
                       'Operating System :: MacOS',
                       'Programming Language :: Python :: 3',
                       ],
          keywords='neuroscience neuroimaging MEG EEG ECoG fNIRS brain',
          project_urls={
              'Documentation': 'https://mne.tools/',
              'Source': 'https://github.com/mne-tools/mne-python/',
              'Tracker': 'https://github.com/mne-tools/mne-python/issues/',
          },
          platforms='any',
          python_requires='>=3.6',
          install_requires=['numpy>=1.11.3', 'scipy>=0.17.1'],
          packages=package_tree('mne'),
          package_data={'mne': [
              op.join('data', '*.sel'),
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
              op.join('datasets', 'sleep_physionet', 'SHA1SUMS'),
              op.join('gui', 'help', '*.json'),
              op.join('html', '*.js'),
              op.join('html', '*.css'),
              op.join('icons', '*.svg'),
              op.join('icons', '*.png'),
              op.join('io', 'artemis123', 'resources', '*.csv'),
              op.join('io', 'edf', 'gdf_encodes.txt')
          ]},
          entry_points={'console_scripts': [
              'mne = mne.commands.utils:main',
          ]})
