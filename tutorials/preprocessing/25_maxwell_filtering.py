# -*- coding: utf-8 -*-
"""
.. _annotations-tutorial:

Signal-space separation
=======================

.. include:: ../../tutorial_links.inc

This tutorial describes how to use Singnal-space separation (SSS), also called
Maxwell filtering, to both reduce environmental noise and compensate for
subject movement in MEG data. As usual we'll start by importing the modules we
need, and loading some example data:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

###############################################################################
# Signal-space separation (SSS), also called Maxwell filtering, is a technique
# based on the physics of electromagnetic fields. SSS separates the measured
# signal into components attributable to sources *inside* the measurement
# volume of the sensor array (the *internal components*), and components
# attributable to sources *outside* the measurement volume (the *external
# components*). The internal and external components are linearly independent,
# so it is possible to simply discard the external components to reduce
# environmental noise.
#
# Like SSP, SSS is a form of projection. Whereas SSP empirically determines a
# noise subspace based on data (empty-room recordings, EOG or ECG activity,
# etc) and projects the measurements onto a subspace orthogonal to the noise,
# SSS mathematically constructs the external and internal subspaces from
# `spherical harmonics`_ and projects the data onto the inside subspace only.
#
# TODO Another great thing is movement compensation...
#
# TODO The main drawback is that it's most thoroughly tested with Elekta
# hardware; results may vary with other MEG manufacturers, see docstring for
# details.
#
#
# Using SSS in MNE-Python
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO

pass

###############################################################################
# References
# ^^^^^^^^^^
#
# Taulu S and Kajola M. (2005). Presentation of electromagnetic multichannel
# data:The signal space separation method. *J Appl Phys* 97, 124905 1-10.
#
# Taulu S and Simola J. (2006). Spatiotemporal signal space separation method
# for rejecting nearby interference in MEG measurements. *Phys Med Biol* 51,
# 1759-1768.
#
#
