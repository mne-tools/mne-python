#! /usr/bin/env python
#
# Copyright (C) 2011 Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>

descr   = """MNE python project for MEG and EEG data analysis."""

import os
import sys

import mne

DISTNAME            = 'mne'
DESCRIPTION         = 'MNE python project for MEG and EEG data analysis'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Alexandre Gramfort'
MAINTAINER_EMAIL    = 'gramfort@nmr.mgh.harvard.edu'
URL                 = 'http://github.com/mne-tools/mne-python'
LICENSE             = 'BSD (3-clause)'
DOWNLOAD_URL        = 'http://github.com/mne-tools/mne-python'
VERSION             = mne.__version__

import setuptools # we are using a setuptools namespace
from numpy.distutils.core import setup

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist', 'bdist_dumb', 'bdist_wininst', 'install_egg_info',
           'build_sphinx', 'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    from setupegg import extra_setuptools_args

# extra_setuptools_args is injected by the setupegg.py script, for
# running the setup with setuptools.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


# if nose available, provide test command
try:
    from nose.commands import nosetests
    cmdclass = extra_setuptools_args.pop('cmdclass', {})
    cmdclass['test'] = nosetests
    cmdclass['nosetests'] = nosetests
    extra_setuptools_args['cmdclass'] = cmdclass
except ImportError:
    pass


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
             ],
         platforms='any',
         packages=['mne', 'mne.tests',
                   'mne.fiff', 'mne.fiff.tests',
                   'mne.datasets', 'mne.datasets.sample',
                   'mne.stats', 'mne.stats.tests',
                   'mne.artifacts', 'mne.artifacts.tests',
                   'mne.minimum_norm', 'mne.minimum_norm.tests',
                   'mne.layouts',
                   'mne.time_frequency', 'mne.time_frequency.tests'],
         **extra_setuptools_args)
