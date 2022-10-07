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
import matplotlib.pyplot as plt
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

# Load raw data and make epochs. For speed, we only use the first 5 events.
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw)[:5]
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2,
                    preload=True)

# We are mostly interested in the beta band since beta desynchronization
# (decrease in power) and a subsequent rebound has been strongly and repeatedly
# linked to movement onset. Let's look at three frequencies in the beta band
# for this example.
freqs = np.array([15, 22.5, 30])

# Use Morlet wavelets to compute sensor-level time-frequency (TFR)
# decomposition for each epoch. We must pass ``output='complex'`` if we wish to
# use this TFR later with a DICS beamformer. We also pass ``average=False`` to
# compute the TFR for each individual epoch.
epochs_tfr = tfr_morlet(epochs, freqs, n_cycles=5, return_itc=False, decim=20,
                        output='complex', average=False)

# %%
# Now, we build a DICS beamformer and project the sensor-level TFR to the
# source level.

# Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
# We are interested in increases in power relative to the baseline period, so
# we will make a separate CSD for just that period as well.
csd = csd_tfr(epochs_tfr, tmin=-1, tmax=1.5)
baseline_csd = csd_tfr(epochs_tfr, tmin=-1, tmax=0)

# Use the CSDs and the forward model to build the DICS beamformer.
fwd = mne.read_forward_solution(fname_fwd)
filters = make_dics(epochs.info, fwd, csd, noise_csd=baseline_csd,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)

# Project the TFR for each epoch to source space
epochs_stcs = apply_dics_tfr_epochs(epochs_tfr, filters, return_generator=True)

# %%
# Finally, let's visualize the source time course estimates. We can see the
# expected activation of the two gyri bordering the central sulcus, the
# primary somatosensory and motor cortices (S1 and M1), this activation
# varies quite a bit trial-to-trial.

fig, axes = plt.subplots(len(freqs), len(events), figsize=(12, 8))
fig.suptitle('Somato dataset beta activation (rebound)')

# iterate over the list of lists (epochs outer list)
for i, stcs in enumerate(epochs_stcs):

    axes[0, i].set_title(f'Epoch {i}')

    for j, stc in enumerate(stcs):  # iterate over frequencies (inner list)

        # At this point, the data is still complex so convert it to power
        stc.data = (stc.data * stc.data.conj()).real

        # Apply a baseline correction
        stc.apply_baseline((-1, 0))

        # Plot the timecourse
        brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.7,
                         hemi='both', views='dorsal',
                         brain_kwargs=dict(show=False),
                         add_data_kwargs=dict(fmin=0, fmid=250, fmax=1000,
                                              colorbar_kwargs=dict(
                                                  label_font_size=10)))
        axes[j, i].imshow(brain.screenshot())
        axes[j, i].set_xticklabels([])
        axes[j, i].set_yticklabels([])
        if i == 0:
            axes[j, i].set_ylabel(f'{freqs[j]} Hz')
        brain.close()

fig.tight_layout()
