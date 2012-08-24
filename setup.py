#! /usr/bin/env python
#
# Copyright (C) 2011 Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>

descr   = """MNE python project for MEG and EEG data analysis."""

import os
import sys

import mne

DISTNAME            = 'mne'
DESCRIPTION         = descr
MAINTAINER          = 'Alexandre Gramfort'
MAINTAINER_EMAIL    = 'gramfort@nmr.mgh.harvard.edu'
URL                 = 'http://martinos.org/mne'
LICENSE             = 'BSD (3-clause)'
DOWNLOAD_URL        = 'http://github.com/mne-tools/mne-python'
VERSION             = mne.__version__

import setuptools # we are using a setuptools namespace
from numpy.distutils.core import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name = DISTNAME,
        maintainer  = MAINTAINER,
        include_package_data = True,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        version = VERSION,
        download_url = DOWNLOAD_URL,
        long_description = open('README.rst').read(),
        zip_safe=False, # the package can run out of an .egg file
        classifiers =
            ['Intended Audience :: Science/Research',
             'Intended Audience :: Developers',
             'License :: OSI Approved',
             'Programming Language :: Python',
             'Topic :: Software Development',
             'Topic :: Scientific/Engineering',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS'
             ],
         platforms='any',
         packages=['mne', 'mne.tests',
                   'mne.fiff', 'mne.fiff.tests',
                   'mne.datasets', 'mne.datasets.sample',
                   'mne.stats', 'mne.stats.tests',
                   'mne.artifacts', 'mne.artifacts.tests',
                   'mne.minimum_norm', 'mne.minimum_norm.tests',
                   'mne.mixed_norm', 'mne.mixed_norm.tests',
                   'mne.layouts',
                   'mne.time_frequency', 'mne.time_frequency.tests',
                   'mne.preprocessing', 'mne.preprocessing.tests',
                   'mne.simulation', 'mne.simulation.tests'],
         package_data={'mne': ['data/mne_analyze.sel']},
         scripts=['bin/mne_clean_eog_ecg.py', 'bin/mne_flash_bem_model.py',
                  'bin/mne_surf2bem.py', 'bin/mne_compute_proj_ecg.py',
                  'bin/mne_compute_proj_eog.py', 'bin/mne_maxfilter.py'])
