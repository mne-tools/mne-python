# -*- coding: utf-8 -*-
"""
EEG source reconstruction with a template MRI
=============================================

This tutorial explains how to perform source reconstruction using
EEG using template MRI subject. We use here the fsaverage brain
provided by freesurder.

.. important:: Source reconstruction without an individual T1 MRI from the
               subject will be less accurate. Do not over interpret
               activity locations which can be off by multiple centimeters.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""  # noqa: E501

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op
import mne
from mne.datasets import sample
from mne.datasets import eegbci
from mne.minimum_norm import make_inverse_operator, apply_inverse

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
trans_fname = op.join(op.dirname(mne.__file__), "data", "fsaverage",
                      "fsaverage-trans.fif")

##############################################################################
# Load the data
# -------------
#
# We use here EEG data from the BCI dataset.

raw_fname, = eegbci.load_data(subject=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True, stim_channel='auto')

# Clean channel names to be able to use a standard 1005 montage
ch_names = [c.replace('.', '') for c in raw.ch_names]
raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

# Read and set the EEG electrode locations
montage = mne.channels.read_montage('standard_1005', ch_names=raw.ch_names,
                                    transform=False)
raw.set_montage(montage)


subject = 'fsaverage'

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, subject=subject, subjects_dir=subjects_dir,
    eeg=['original', 'projected'], trans=None
)

##############################################################################
# Setup source space and compute forward
# --------------------------------------

src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
print(src)

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw.info, trans=trans_fname, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=2)
print(fwd)

##############################################################################
# Epoch data
# ----------

tmin, tmax = -1, 4  # define epochs around events (in s)
event_ids = dict(hands=2, feet=3)  # map event IDs to tasks
events, _ = mne.events_from_annotations(raw)
raw.filter(1., 40.)
raw.set_eeg_reference(projection=True)
epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=('eeg'), baseline=None, preload=True)

evoked = epochs.average()

##############################################################################
# Compute dSPM source estimates
# -----------------------------

# Compute noise cov
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank='full')

# make an MEG inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)


method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

###############################################################################
# Now plot the source estimates
# -----------------------------

vertno_max, time_max = stc.get_peak(hemi='rh')

subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir, views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
