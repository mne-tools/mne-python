#! /usr/bin/env python
#
# Copyright (C) 2011 Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>

descr   = """MNE python project for MEG and EEG data analysis."""

import os


DISTNAME            = 'mne'
DESCRIPTION         = 'MNE python project for MEG and EEG data analysis'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Alexandre Gramfort'
MAINTAINER_EMAIL    = 'gramfort@nmr.mgh.harvard.edu'
URL                 = 'http://github.com/mne/mne'
LICENSE             = 'BSD'
DOWNLOAD_URL        = 'http://github.com/mne/mne'
VERSION             = '0.1.git'

import setuptools # we are using a setuptools namespace
from numpy.distutils.core import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    setup(name = DISTNAME,
        maintainer  = MAINTAINER,
        include_package_data = True,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        version = VERSION,
        download_url = DOWNLOAD_URL,
        long_description = LONG_DESCRIPTION,
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
             ]
    )
