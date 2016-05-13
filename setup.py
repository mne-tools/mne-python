#! /usr/bin/env python
#

# Copyright (C) 2011-2014 Alexandre Gramfort
# <alexandre.gramfort@telecom-paristech.fr>

import os
from os import path as op

import setuptools  # noqa; we are using a setuptools namespace
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
MAINTAINER_EMAIL = 'alexandre.gramfort@telecom-paristech.fr'
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
                    'mne.commands', 'mne.commands.tests',
                    'mne.connectivity', 'mne.connectivity.tests',
                    'mne.data',
                    'mne.datasets',
                    'mne.datasets.eegbci',
                    'mne.datasets._fake',
                    'mne.datasets.megsim',
                    'mne.datasets.misc',
                    'mne.datasets.sample',
                    'mne.datasets.somato',
                    'mne.datasets.spm_face',
                    'mne.datasets.brainstorm',
                    'mne.datasets.testing',
                    'mne.datasets.tests',
                    'mne.externals',
                    'mne.externals.h5io',
                    'mne.externals.tempita',
                    'mne.io', 'mne.io.tests',
                    'mne.io.array', 'mne.io.array.tests',
                    'mne.io.brainvision', 'mne.io.brainvision.tests',
                    'mne.io.bti', 'mne.io.bti.tests',
                    'mne.io.cnt', 'mne.io.cnt.tests',
                    'mne.io.ctf', 'mne.io.ctf.tests',
                    'mne.io.edf', 'mne.io.edf.tests',
                    'mne.io.egi', 'mne.io.egi.tests',
                    'mne.io.fiff', 'mne.io.fiff.tests',
                    'mne.io.kit', 'mne.io.kit.tests',
                    'mne.io.nicolet', 'mne.io.nicolet.tests',
                    'mne.io.eeglab', 'mne.io.eeglab',
                    'mne.forward', 'mne.forward.tests',
                    'mne.viz', 'mne.viz.tests',
                    'mne.gui', 'mne.gui.tests',
                    'mne.minimum_norm', 'mne.minimum_norm.tests',
                    'mne.inverse_sparse', 'mne.inverse_sparse.tests',
                    'mne.preprocessing', 'mne.preprocessing.tests',
                    'mne.simulation', 'mne.simulation.tests',
                    'mne.tests',
                    'mne.stats', 'mne.stats.tests',
                    'mne.time_frequency', 'mne.time_frequency.tests',
                    'mne.realtime', 'mne.realtime.tests',
                    'mne.decoding', 'mne.decoding.tests',
                    'mne.commands',
                    'mne.channels', 'mne.channels.tests'],
          package_data={'mne': [op.join('data', '*.sel'),
                                op.join('data', 'icos.fif.gz'),
                                op.join('data', 'coil_def*.dat'),
                                op.join('data', 'helmets', '*.fif.gz'),
                                op.join('data', 'FreeSurferColorLUT.txt'),
                                op.join('data', 'image', '*gif'),
                                op.join('data', 'image', '*lout'),
                                op.join('channels', 'data', 'layouts', '*.lout'),
                                op.join('channels', 'data', 'layouts', '*.lay'),
                                op.join('channels', 'data', 'montages', '*.sfp'),
                                op.join('channels', 'data', 'montages', '*.txt'),
                                op.join('channels', 'data', 'montages', '*.elc'),
                                op.join('channels', 'data', 'neighbors', '*.mat'),
                                op.join('gui', 'help', '*.json'),
                                op.join('html', '*.js'),
                                op.join('html', '*.css')]},
          scripts=['bin/mne'])
