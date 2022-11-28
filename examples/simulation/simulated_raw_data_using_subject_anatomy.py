# -*- coding: utf-8 -*-
"""
.. _ex-sim-raw-sub:

=======================================
Simulate raw data using subject anatomy
=======================================

This example illustrates how to generate source estimates and simulate raw data
using subject anatomy with the :class:`mne.simulation.SourceSimulator` class.
Once the raw data is simulated, generated source estimates are reconstructed
using dynamic statistical parametric mapping (dSPM) inverse operator.
"""

# Author: Ivana Kojcic <ivana.kojcic@gmail.com>
#         Eric Larson <larson.eric.d@gmail.com>
#         Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#         Samuel Deslauriers-Gauthier <sam.deslauriers@gmail.com>

# License: BSD-3-Clause

# %%

import numpy as np

import mne
from mne.datasets import sample

print(__doc__)

# %%
# In this example, raw data will be simulated for the sample subject, so its
# information needs to be loaded. This step will download the data if it not
# already on your machine. Subjects directory is also set so it doesn't need
# to be given to functions.

data_path = sample.data_path()
subjects_dir = data_path / 'subjects'
subject = 'sample'
meg_path = data_path / 'MEG' / subject

# %%
# First, we get an info structure from the sample subject.

fname_info = meg_path / 'sample_audvis_raw.fif'
info = mne.io.read_info(fname_info)
tstep = 1 / info['sfreq']

# %%
# To simulate sources, we also need a source space. It can be obtained from the
# forward solution of the sample subject.

fwd_fname = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

# %%
# To simulate raw data, we need to define when the activity occurs using events
# matrix and specify the IDs of each event.
# Noise covariance matrix also needs to be defined.
# Here, both are loaded from the sample dataset, but they can also be specified
# by the user.

fname_event = meg_path / 'sample_audvis_raw-eve.fif'
fname_cov = meg_path / 'sample_audvis-cov.fif'

events = mne.read_events(fname_event)
noise_cov = mne.read_cov(fname_cov)

# Standard sample event IDs. These values will correspond to the third column
# in the events matrix.
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}


# Take only a few events for speed
events = events[:80]

# %%
# In order to simulate source time courses, labels of desired active regions
# need to be specified for each of the 4 simulation conditions.
# Make a dictionary that maps conditions to activation strengths within
# aparc.a2009s :footcite:`DestrieuxEtAl2010` labels.
# In the aparc.a2009s parcellation:
#
# - 'G_temp_sup-G_T_transv' is the label for primary auditory area
# - 'S_calcarine' is the label for primary visual area
#
# In each of the 4 conditions, only the primary area is activated. This means
# that during the activations of auditory areas, there are no activations in
# visual areas and vice versa.
# Moreover, for each condition, contralateral region is more active (here, 2
# times more) than the ipsilateral.

activations = {
    'auditory/left':
        [('G_temp_sup-G_T_transv-lh', 30),          # label, activation (nAm)
         ('G_temp_sup-G_T_transv-rh', 60)],
    'auditory/right':
        [('G_temp_sup-G_T_transv-lh', 60),
         ('G_temp_sup-G_T_transv-rh', 30)],
    'visual/left':
        [('S_calcarine-lh', 30),
         ('S_calcarine-rh', 60)],
    'visual/right':
        [('S_calcarine-lh', 60),
         ('S_calcarine-rh', 30)],
}

annot = 'aparc.a2009s'

# Load the 4 necessary label names.
label_names = sorted(set(activation[0]
                         for activation_list in activations.values()
                         for activation in activation_list))
region_names = list(activations.keys())

# %%
# Create simulated source activity
# --------------------------------
#
# Generate source time courses for each region. In this example, we want to
# simulate source activity for a single condition at a time. Therefore, each
# evoked response will be parametrized by latency and duration.


def data_fun(times, latency, duration):
    """Function to generate source time courses for evoked responses,
    parametrized by latency and duration."""
    f = 15  # oscillating frequency, beta band [Hz]
    sigma = 0.375 * duration
    sinusoid = np.sin(2 * np.pi * f * (times - latency))
    gf = np.exp(- (times - latency - (sigma / 4.) * rng.rand(1)) ** 2 /
                (2 * (sigma ** 2)))
    return 1e-9 * sinusoid * gf


# %%
# Here, :class:`~mne.simulation.SourceSimulator` is used, which allows to
# specify where (label), what (source_time_series), and when (events) event
# type will occur.
#
# We will add data for 4 areas, each of which contains 2 labels. Since add_data
# method accepts 1 label per call, it will be called 2 times per area.
#
# Evoked responses are generated such that the main component peaks at 100ms
# with a duration of around 30ms, which first appears in the contralateral
# cortex. This is followed by a response in the ipsilateral cortex with a peak
# about 15ms after. The amplitude of the activations will be 2 times higher in
# the contralateral region, as explained before.
#
# When the activity occurs is defined using events. In this case, they are
# taken from the original raw data. The first column is the sample of the
# event, the second is not used. The third one is the event id, which is
# different for each of the 4 areas.

times = np.arange(150, dtype=np.float64) / info['sfreq']
duration = 0.03
rng = np.random.RandomState(7)
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)

for region_id, region_name in enumerate(region_names, 1):
    events_tmp = events[np.where(events[:, 2] == region_id)[0], :]
    for i in range(2):
        label_name = activations[region_name][i][0]
        label_tmp = mne.read_labels_from_annot(subject, annot,
                                               subjects_dir=subjects_dir,
                                               regexp=label_name,
                                               verbose=False)
        label_tmp = label_tmp[0]
        amplitude_tmp = activations[region_name][i][1]
        if region_name.split('/')[1][0] == label_tmp.hemi[0]:
            latency_tmp = 0.115
        else:
            latency_tmp = 0.1
        wf_tmp = data_fun(times, latency_tmp, duration)
        source_simulator.add_data(label_tmp,
                                  amplitude_tmp * wf_tmp,
                                  events_tmp)

# To obtain a SourceEstimate object, we need to use `get_stc()` method of
# SourceSimulator class.
stc_data = source_simulator.get_stc()

# %%
# Simulate raw data
# -----------------
#
# Project the source time series to sensor space. Three types of noise will be
# added to the simulated raw data:
#
# - multivariate Gaussian noise obtained from the noise covariance from the
#   sample data
# - blink (EOG) noise
# - ECG noise
#
# The :class:`~mne.simulation.SourceSimulator` can be given directly to the
# :func:`~mne.simulation.simulate_raw` function.

raw_sim = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
raw_sim.set_eeg_reference(projection=True)

mne.simulation.add_noise(raw_sim, cov=noise_cov, random_state=0)
mne.simulation.add_eog(raw_sim, random_state=0)
mne.simulation.add_ecg(raw_sim, random_state=0)

# Plot original and simulated raw data.
raw_sim.plot(title='Simulated raw data')

# %%
# Extract epochs and compute evoked responsses
# --------------------------------------------
#

epochs = mne.Epochs(raw_sim, events, event_id, tmin=-0.2, tmax=0.3,
                    baseline=(None, 0))
evoked_aud_left = epochs['auditory/left'].average()
evoked_vis_right = epochs['visual/right'].average()

# Visualize the evoked data
evoked_aud_left.plot(spatial_colors=True)
evoked_vis_right.plot(spatial_colors=True)

# %%
# Reconstruct simulated source time courses using dSPM inverse operator
# ---------------------------------------------------------------------
#
# Here, source time courses for auditory and visual areas are reconstructed
# separately and their difference is shown. This was done merely for better
# visual representation of source reconstruction.
# As expected, when high activations appear in primary auditory areas, primary
# visual areas will have low activations and vice versa.

method, lambda2 = 'dSPM', 1. / 9.
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc_aud = mne.minimum_norm.apply_inverse(
    evoked_aud_left, inv, lambda2, method)
stc_vis = mne.minimum_norm.apply_inverse(
    evoked_vis_right, inv, lambda2, method)
stc_diff = stc_aud - stc_vis

brain = stc_diff.plot(subjects_dir=subjects_dir, initial_time=0.1,
                      hemi='split', views=['lat', 'med'])

# %%
# References
# ----------
# .. footbibliography::
