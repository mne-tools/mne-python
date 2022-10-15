# -*- coding: utf-8 -*-
"""
.. _tut-tfr_stc:

====================================
Time-frequency source reconstruction
====================================

This tutorial shows how to use source estimation in MNE for time-frequency
resolved data. And, it shows how to visualize time-frequency source
estimations using a graphical user interface.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

import numpy as np
import mne
from mne.datasets import somato

# %%
# Load sensor data and compute time-frequency decomposition
# ---------------------------------------------------------
# We will use the sample data set for this tutorial and reconstruct source
# activity on the trials with a left auditory stimulus.
#
# .. note:: You would want to use preprocessing methods such as epoch
#           rejection and ica artifact rejection (:ref:`tut-reject-data-spans`,
#           :ref:`tut-artifact-ica` and :ref:`ex-muscle-ica`) to clean the
#           data for a real analysis. Here we will set extra-strict amplitude
#           rejection thresholds as a shortcut for only selecting clean trials.

data_path = somato.data_path()
subject = '01'
task = 'somato'
subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'
raw_fname = (data_path / f'sub-{subject}' / 'meg' /
             f'sub-{subject}_task-{task}_meg.fif')
# Read the raw data
raw = mne.io.read_raw_fif(raw_fname)
raw.info['bads'] = ['MEG 2443']  # bad MEG channel

# Set up the epoching
event_id = 1  # those are the trials with left-ear auditory stimuli
tmin, tmax = -0.5, 0.5
events = mne.find_events(raw)

# create epochs, leave room for cutting out edge artifact
epochs = mne.Epochs(raw, events[events[:, 2] == event_id],
                    tmin=tmin, tmax=tmax,
                    baseline=(None, 0), preload=True, proj=False)
# select first ten for speed, use whole thing later
epochs = epochs[:10]
# set average reference
# epochs.set_eeg_reference('average', projection=True)
# compute time-frequency decomposition, focus on alpha and beta oscillations
freqs = np.logspace(*np.log10([8, 35]), num=6)

# time-frequency decomposition
epochs_tfr = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=freqs / 2, return_itc=False,
    average=False, output='complex')

# plot data, there's a lot so you can't see much but we'll see it better
# later in the tutorial!
average_tfr = epochs_tfr.average()
average_tfr.plot_topo(title='Average Amplitude')

# %%
# The forward model
# -----------------
# Next, we'll need our forward model of how sensor data relates to data at
# each of the source vertices that we've chosen, see :ref:`tut-forward` for
# more information.

surface = subjects_dir / subject / 'bem' / 'inner_skull.surf'
vol_src = mne.setup_volume_source_space(
    subject=subject, subjects_dir=subjects_dir, surface=surface,
    pos=10, add_interpolator=False)  # just for speed!

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject=subject, ico=3,  # just for speed
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# load the forward model just for the head->mri transform
fname_fwd = (data_path / 'derivatives' / f'sub-{subject}' /
             f'sub-{subject}_task-{task}-fwd.fif')
forward_tmp = mne.read_forward_solution(fname_fwd)
trans = forward_tmp['info']['mri_head_t']
forward = mne.make_forward_solution(
    raw.info, trans=trans, src=vol_src, bem=bem, meg=True, eeg=True,
    mindist=5.0, n_jobs=1, verbose=True)

# %%
# Computing the covariance matrices
# ---------------------------------
# In order to compute the source estimate, we need a covariance matrix to
# whiten our data, in this case, the time-frequency analog; a cross-spectral
# density matrix, see :ref:`ex-csd-matrix` for more information.

data_csd = mne.time_frequency.csd_tfr(epochs_tfr, tmin=0.01, tmax=tmax)
noise_csd = mne.time_frequency.csd_tfr(epochs_tfr, tmin=tmin, tmax=0)

data_csd.plot(epochs_tfr.info)

# %%
# Use minimum norm estimation
# ----------------------------
# Finally, we'll read the volumetric source space and use it to create

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'  # use dSPM method (could also be dSPM or sLORETA)

inverse_operator = list()
for freq_idx in range(epochs_tfr.freqs.size):
    noise_cov = noise_csd.get_data(index=freq_idx, as_cov=True)
    noise_cov['data'] = noise_cov['data'].real  # only normalize by real
    inverse_operator.append(mne.minimum_norm.make_inverse_operator(
        epochs_tfr.info, forward, noise_cov))

# decimate for speed and memory usage in this tutorial, you probably
# shouldn't decimate so much in a real analysis
epochs_tfr.decimate(decim=20)

stcs = mne.minimum_norm.apply_inverse_tfr_epochs(
    epochs_tfr, inverse_operator, lambda2, method=method,
    pick_ori='vector')

# note, here frequencies are the outer list, opposite of the beamformer
# here, we used pick_ori='vector' so we have an orientation dimension

# compute power, take the average over epochs and cast to integers to save
# memory, the GUI can also handle complex data across epochs if your
# computer has enough RAM but this really lowers the memory usage
data = np.array([(np.mean(
    [(stc.data * stc.data.conj()).real for stc in tfr_stcs],
    axis=0, keepdims=True) * 1e32).astype(np.uint64) for tfr_stcs in stcs])
data = data.transpose((1, 2, 3, 0, 4))  # move frequencies to penultimate

# gain normalize
data = data // data.mean(axis=-1, keepdims=True)

viewer = mne.gui.view_stc(data, subject=subject, subjects_dir=subjects_dir,
                          src=vol_src, inst=epochs_tfr)
viewer.go_to_max()  # show the maximum intensity source vertex
