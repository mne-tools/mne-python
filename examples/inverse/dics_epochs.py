# -*- coding: utf-8 -*-
"""
.. _ex-inverse-dics-epochs:

=======================================================================
Compute source level time-frequency timecourses using a DICS beamformer
=======================================================================

In this example, a Dynamic Imaging of Coherent Sources (DICS)
:footcite:`GrossEtAl2001` beamformer is used to transform sensor-level
time-frequency objects to the source level. We will look at the event-related
synchronization (ERS) of beta band activity in the :ref:`somato dataset
<somato-dataset>`.
"""
# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import numpy as np
import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet, csd_tfr
from mne.beamformer import make_dics, apply_dics_tfr_epochs

print(__doc__)

# %%
# Organize the data that we will use for this example.

data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = (data_path / f'sub-{subject}' / 'meg' /
             f'sub-{subject}_task-{task}_meg.fif')
fname_fwd = (data_path / 'derivatives' / f'sub-{subject}' /
             f'sub-{subject}_task-{task}-fwd.fif')
subjects_dir = data_path / 'derivatives' / 'freesurfer' / 'subjects'

# %%
# First, we load the data and compute for each epoch the time-frequency
# decomposition in sensor space.

# Load raw data and make epochs.
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1, tmax=2.5,
                    reject=dict(grad=5000e-13,  # unit: T / m (gradiometers)
                                mag=5e-12,      # unit: T (magnetometers)
                                eog=250e-6,    # unit: V (EOG channels)
                                ), preload=True)
epochs = epochs[:10]  # just for speed of execution for the tutorial

# We are mostly interested in the beta band since it has been shown to be
# active for somatosensory stimulation
freqs = np.linspace(13, 31, 5)

# Use Morlet wavelets to compute sensor-level time-frequency (TFR)
# decomposition for each epoch. We must pass ``output='complex'`` if we wish to
# use this TFR later with a DICS beamformer. We also pass ``average=False`` to
# compute the TFR for each individual epoch.
epochs_tfr = tfr_morlet(epochs, freqs, n_cycles=5, return_itc=False,
                        output='complex', average=False)

# crop either side to use a buffer to remove edge artifact
epochs_tfr.crop(tmin=-0.5, tmax=2)

# %%
# Now, we build a DICS beamformer and project the sensor-level TFR to the
# source level.

# Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
# We are interested in increases in power relative to the baseline period, so
# we will make a separate CSD for just that period as well.
csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=2)
baseline_csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=-0.1)

# use the CSDs and the forward model to build the DICS beamformer
fwd = mne.read_forward_solution(fname_fwd)

# compute scalar DICS beamfomer
filters = make_dics(epochs.info, fwd, csd, noise_csd=baseline_csd,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)

# project the TFR for each epoch to source space
epochs_stcs = apply_dics_tfr_epochs(
    epochs_tfr, filters, return_generator=True)

# average across frequencies and epochs
data = np.zeros((fwd['nsource'], epochs_tfr.times.size))
for epoch_stcs in epochs_stcs:
    for stc in epoch_stcs:
        data += (stc.data * np.conj(stc.data)).real

stc.data = data / len(epochs) / len(freqs)

# apply a baseline correction
stc.apply_baseline((-0.5, -0.1))

# %%
# Let's visualize the source time course estimate. We can see the
# expected activation of the two gyri bordering the central sulcus, the
# primary somatosensory and motor cortices (S1 and M1).

fmax = 4500
brain = stc.plot(
    subjects_dir=subjects_dir,
    hemi='both',
    views='dorsal',
    initial_time=0.55,
    brain_kwargs=dict(show=False),
    add_data_kwargs=dict(fmin=fmax / 10, fmid=fmax / 2, fmax=fmax,
                         scale_factor=0.0001,
                         colorbar_kwargs=dict(label_font_size=10))
)

# You can save a movie like the one on our documentation website with:
# brain.save_movie(tmin=0.55, tmax=1.5, interpolation='linear',
#                  time_viewer=True)
