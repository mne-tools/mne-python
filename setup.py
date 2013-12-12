#! /usr/bin/env python
#
# Copyright (C) 2011-2013 Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>

import os

import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup

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
MAINTAINER_EMAIL = 'gramfort@nmr.mgh.harvard.edu'
URL = 'http://martinos.org/mne'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-python'
VERSION = version


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
          packages=['mne', 'mne.tests',
                    'mne.beamformer', 'mne.beamformer.tests',
                    'mne.connectivity', 'mne.connectivity.tests',
                    'mne.data',
                    'mne.datasets',
                    'mne.datasets.sample',
                    'mne.datasets.megsim',
                    'mne.datasets.spm_face',
                    'mne.externals',
                    'mne.fiff', 'mne.fiff.tests',
                    'mne.fiff.bti', 'mne.fiff.bti.tests',
                    'mne.fiff.kit', 'mne.fiff.kit.tests',
                    'mne.fiff.edf', 'mne.fiff.edf.tests',
                    'mne.fiff.brainvision', 'mne.fiff.brainvision.tests',
                    'mne.forward', 'mne.forward.tests',
                    'mne.gui', 'mne.gui.tests',
                    'mne.layouts', 'mne.layouts.tests',
                    'mne.minimum_norm', 'mne.minimum_norm.tests',
                    'mne.mixed_norm',
                    'mne.inverse_sparse', 'mne.inverse_sparse.tests',
                    'mne.preprocessing', 'mne.preprocessing.tests',
                    'mne.simulation', 'mne.simulation.tests',
                    'mne.tests',
                    'mne.stats', 'mne.stats.tests',
                    'mne.time_frequency', 'mne.time_frequency.tests',
                    'mne.realtime', 'mne.realtime.tests',
                    'mne.decoding', 'mne.decoding.tests',
                    'mne.commands', 'mne.externals'],
          package_data={'mne': ['data/*.sel',
                                'data/icos.fif.gz',
                                'data/coil_def.dat',
                                'layouts/*.lout',
                                'layouts/*.lay']},
          scripts=['bin/mne'])
