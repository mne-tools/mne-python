# -*- coding: utf-8 -*-
"""
EEG forward operator with a template MRI
========================================

This tutorial explains how to compute the forward operator from EEG data
using the standard template MRI subject ``fsaverage``.

.. important:: Source reconstruction without an individual T1 MRI from the
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
from mne.datasets import set_montage_coreg_path, fetch_fsaverage, sample

# Convenience function to set ``subjects_dir`` default value for users who
# only ever plan to do montage-based coreg with fsaverage.
set_montage_coreg_path()

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
# The files live in:
subject = 'fsaverage'
trans = op.join(fs_dir, 'bem', 'fsaverage-trans.fif')
src = op.join(fs_dir, 'bem', 'fsaverage-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

##############################################################################
# Load the data
# -------------
# We use here sample data limited to EEG channels, and replace our digitized
# locations with those from a montage.

data_path = sample.data_path(verbose=True)
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
raw.info['bads'] = ['EEG 053']
raw.load_data().pick_types(meg=False, eeg=True, eog=True, stim=True)
montage = mne.channels.read_montage('mgh60', transform=True)
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans, dig=True)

##############################################################################
# Get ERP
# -------
# Average over the auditory condition to get an ERP.

events = mne.find_events(raw)
event_id = dict(aud_l=1, aud_r=2)
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, proj=True,
                    baseline=(None, 0), reject=dict(eog=150e-6))
evoked = epochs.average()
fig = evoked.plot()
max_time = evoked.get_peak()[1]
fig.axes[0].axvline(max_time, color='g', ls=':')

##############################################################################
# Setup source space and compute forward
# --------------------------------------

fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)

###############################################################################
# Compute source activation
# -------------------------

cov = mne.compute_covariance(epochs, tmax=0.)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov)
print(inv)
stc = mne.minimum_norm.apply_inverse(evoked, inv)
brain = stc.plot(initial_time=max_time)
