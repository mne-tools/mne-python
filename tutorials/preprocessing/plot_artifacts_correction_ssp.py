"""
.. _tut-artifact-ssp:

Artifact Correction with SSP
============================

This tutorial explains how to estimate Signal Subspace Projectors (SSP)
for correction of ECG and EOG artifacts.

See :ref:`sphx_glr_auto_examples_io_plot_read_proj.py` for how to read
and visualize already present SSP projection vectors.

"""
import numpy as np

import mne
from mne.datasets import sample
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

# getting some data ready
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)

##############################################################################
# Compute SSP projections
# -----------------------
#
# First let's do ECG.

projs, events = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=True)
print(projs)

ecg_projs = projs[-2:]
mne.viz.plot_projs_topomap(ecg_projs)

###############################################################################
# Now let's do EOG. Here we compute an EEG projector, and need to pass
# the measurement info so the topomap coordinates can be created.

projs, events = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, average=True)
print(projs)

eog_projs = projs[-3:]
mne.viz.plot_projs_topomap(eog_projs, info=raw.info)

##############################################################################
# Apply SSP projections
# ---------------------
#
# MNE is handling projections at the level of the info,
# so to register them populate the list that you find in the 'proj' field

raw.info['projs'] += eog_projs + ecg_projs

#############################################################################
# Yes this was it. Now MNE will apply the projs on demand at any later stage,
# so watch out for proj parameters in functions or to it explicitly
# with the ``.apply_proj`` method

#############################################################################
# Demonstrate SSP cleaning on some evoked data
# --------------------------------------------

events = mne.find_events(raw, stim_channel='STI 014')
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
# this can be highly data dependent
event_id = {'auditory/left': 1}

epochs_no_proj = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                            proj=False, baseline=(None, 0), reject=reject)
epochs_no_proj.average().plot(spatial_colors=True, time_unit='s')


epochs_proj = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, proj=True,
                         baseline=(None, 0), reject=reject)
epochs_proj.average().plot(spatial_colors=True, time_unit='s')

##############################################################################
# Looks cool right? It is however often not clear how many components you
# should take and unfortunately this can have bad consequences as can be seen
# interactively using the delayed SSP mode:

evoked = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                    proj='delayed', baseline=(None, 0),
                    reject=reject).average()

# set time instants in seconds (from 50 to 150ms in a step of 10ms)
times = np.arange(0.05, 0.15, 0.01)

fig = evoked.plot_topomap(times, proj='interactive', time_unit='s')

##############################################################################
# now you should see checkboxes. Remove a few SSP and see how the auditory
# pattern suddenly drops off
