# -*- coding: utf-8 -*-
"""
.. _tut-eeg-fsaverage-source-modeling:

EEG forward operator with a template MRI
========================================

This tutorial explains how to compute the forward operator from EEG data
using the standard template MRI subject ``fsaverage``.

.. caution:: Source reconstruction without an individual T1 MRI from the
             subject will be less accurate. Do not over interpret
             activity locations which can be off by multiple centimeters.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

##############################################################################
# Load the data
# -------------
#
# We use here EEG data from the BCI dataset.
#
# .. note:: See :ref:`plot_montage` to view all the standard EEG montages
#           available in MNE-Python.

raw_fname, = eegbci.load_data(subject=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)

# Clean channel names to be able to use a standard 1005 montage
new_names = dict(
    (ch_name,
     ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
    for ch_name in raw.ch_names)
raw.rename_channels(new_names)

# Read and set the EEG electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

##############################################################################
# Setup source space and compute forward
# --------------------------------------

fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)

# Use fwd to compute the sensitivity map for illustration purposes
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
brain = eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
                     clim=dict(lims=[5, 50, 100]))
