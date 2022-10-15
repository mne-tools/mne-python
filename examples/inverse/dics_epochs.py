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
events = mne.find_events(raw)[:3]
epochs = mne.Epochs(raw, events, event_id=1, tmin=-1.5, tmax=2,
                    preload=True)

# We are mostly interested in the beta band since beta desynchronization
# (decrease in power) and a subsequent rebound has been strongly and repeatedly
# linked to movement onset. Let's look at three frequencies in the beta band
# for this example.
freqs = np.array([15, 30])

# Use Morlet wavelets to compute sensor-level time-frequency (TFR)
# decomposition for each epoch. We must pass ``output='complex'`` if we wish to
# use this TFR later with a DICS beamformer. We also pass ``average=False`` to
# compute the TFR for each individual epoch.
epochs_tfr = tfr_morlet(epochs, freqs, n_cycles=5, return_itc=False,
                        output='complex', average=False)

# %%
# Now, we build a DICS beamformer and project the sensor-level TFR to the
# source level.

# Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
# We are interested in increases in power relative to the baseline period, so
# we will make a separate CSD for just that period as well.
csd = csd_tfr(epochs_tfr, tmin=-1, tmax=1.5)
baseline_csd = csd_tfr(epochs_tfr, tmin=-1, tmax=0)

# use the CSDs and the forward model to build the DICS beamformer
fwd = mne.read_forward_solution(fname_fwd)

# compute vector solution
filters = make_dics(epochs.info, fwd, csd, noise_csd=baseline_csd,
                    pick_ori='vector', reduce_rank=True, real_filter=True)

# project the TFR for each epoch to source space
epochs_stcs = apply_dics_tfr_epochs(
    epochs_tfr, filters, return_generator=True)

# %%
# Let's visualize the source time course estimates. We can see the
# expected activation of the two gyri bordering the central sulcus, the
# primary somatosensory and motor cortices (S1 and M1), this activation
# varies quite a bit trial-to-trial.

fig, axes = plt.subplots(len(freqs), len(events), figsize=(12, 8))
fig.suptitle('Somato dataset beta activation')

# iterate over the list of lists (epochs outer list)
for i, stcs in enumerate(epochs_stcs):

    axes[0, i].set_title(f'Epoch {i}')

    # iterate over frequencies (inner list)
    for j, stc in enumerate(stcs):

        # convert from complex time-frequency to power
        stc.data = (stc.data * np.conj(stc.data)).real

        # apply a baseline correction
        stc.apply_baseline((-1, 0))

        # plot the timecourse direction
        brain = stc.plot(
            subjects_dir=subjects_dir,
            hemi='both',
            initial_time=0.7,
            brain_kwargs=dict(show=False),
            add_data_kwargs=dict(fmin=2000, fmid=5000, fmax=10000,
                                 scale_factor=0.001,
                                 colorbar_kwargs=dict(label_font_size=10))
        )
        brain.show_view(azimuth=-40, elevation=35, distance=300,
                        focalpoint=(0, 0, 0))
        axes[j, i].imshow(brain.screenshot())
        brain.close()

        axes[j, i].set_xticklabels([])
        axes[j, i].set_yticklabels([])
        if i == 0:
            axes[j, i].set_ylabel(f'{freqs[j]} Hz')

fig.tight_layout()

# %%
# We can also view the phase for each time-frequency source time course.
# Before, we computed the the direction of the activity but this would
# give us three independent phase estimates for the
# ``x``, ``y`` and ``z`` directions, which we don't want. Instead, we'll
# use the phase from the maximum power orientation which is a more
# sensible way to determine a single phase at each frequency over time.

# use the maximum power orientation for phase
filters = make_dics(epochs.info, fwd, csd, noise_csd=baseline_csd,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)

# project the TFR for each epoch to source space
epochs_stcs = apply_dics_tfr_epochs(
    epochs_tfr, filters, return_generator=True)

# select one stc for an example
stc = next(epochs_stcs)[0]

# compute mask to zero out vertices with low power
mask = (stc.data * np.conj(stc.data)).real < 5000

# compute phase, add offset for plotting
stc.data = np.angle(stc.data) + 10 + np.pi

# apply the mask
stc.data[mask] = 0

# plot the timecourse phase
brain = stc.plot(
    subjects_dir=subjects_dir,
    hemi='both',
    views='dorsal',
    initial_time=0.875,
    brain_kwargs=dict(surf='inflated'),
    add_data_kwargs=dict(fmin=10, fmid=10 + np.pi, fmax=10 + 2 * np.pi,
                         colormap='mne')
)
brain.show_view(azimuth=-40, elevation=35, distance=300,
                focalpoint=(0, 0, 0))

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=30, tmin=0.875, tmax=0.95,
#                  interpolation='linear', time_viewer=True)
