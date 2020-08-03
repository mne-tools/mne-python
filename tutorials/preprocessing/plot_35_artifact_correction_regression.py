# -*- coding: utf-8 -*-
"""
.. _tut-artifact-regression:

EOG correction using regression
===============================

This tutorial covers removal of artifacts using regression as in
:footcite:`GrattonEtAl1983`.

.. contents:: Page contents
   :local:
   :depth: 2

We begin as always by importing the necessary Python modules and loading some
data. In this case we use data from :ref:`brainstorm <tut-brainstorm-auditory>`
because it has both EOG and ECG electrodes. We then crop it to 60 seconds,
set some channel information, then process the data.

That there are other corrections that are useful for this dataset that we will
not apply here -- see :ref:`tut-brainstorm-auditory` for more information.
"""

import os.path as op
import numpy as np
import mne

data_path = mne.datasets.brainstorm.bst_auditory.data_path()
raw_fname = op.join(data_path, 'MEG', 'bst_auditory', 'S01_AEF_20131218_01.ds')
raw = mne.io.read_raw_ctf(raw_fname).crop(0, 60)
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
raw.pick(['meg', 'stim', 'misc', 'eog', 'ecg']).load_data()
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

###############################################################################
# Removing artifacts by regression
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Artifacts whose time waveforms are accurately reflected by some reference
# signal can be removed by regression. Two examples of this are EOG artifacts
# captured by EOG channels, and ECG signals captured by ECG electrodes.
#
#
# EOG
# ~~~
#
# First let's epoch our data the normal way, using just event type 1 (standard)
# for simplicity. Event timing is adjusted by comparing the trigger times on
# detected sound onsets on channel UADC001-4408.

events = mne.find_events(raw, stim_channel='UPPT001')
sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
min_diff = int(0.5 * raw.info['sfreq'])
diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
onsets = onsets[diffs > min_diff]
assert len(onsets) == len(events)
events[:, 0] = onsets
epochs = mne.Epochs(raw, events, event_id=1, preload=True)

###############################################################################
# Now let's regress our EOG signal and plot the original and processed data.
# We do this by estimating the regression coefficients on data without the
# evoked response, then use those coefficients to remove the EOG signal:

plot_picks = ['meg', 'eog', 'ecg']
fig = epochs.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('Auditory epochs')
fig.tight_layout()

epochs_no_ave = epochs.copy().subtract_evoked()
_, betas = mne.preprocessing.regress(epochs_no_ave)
epochs_clean, _ = mne.preprocessing.regress(epochs, betas=betas)
fig = epochs_clean.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('Auditory epochs, EOG regressed')
fig.tight_layout()

###############################################################################
# The effect is subtle in these evoked data. It's clearer if we create epochs
# around our (autodetected) EOG events and plot the average across those
# epochs:

eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_epochs.apply_baseline((None, None))
raw.plot(events=eog_epochs.events)
fig = eog_epochs.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('EOG epochs')
fig.tight_layout()

eog_epochs_clean, _ = mne.preprocessing.regress(eog_epochs)
fig = eog_epochs_clean.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('EOG epochs, EOG regressed')
fig.tight_layout()

###############################################################################
# ECG
# ~~~
# We can follow the same process with ECG:

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.apply_baseline((None, None))
raw.plot(events=ecg_epochs.events)
fig = ecg_epochs.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('ECG epochs')
fig.tight_layout()

###############################################################################
# But it does not remove as much of the signal, even from the reference
# epochs, likely because the ECG signal is a rotating dipole, and the ECG
# electrode time waveform does not capture the same effective time waveform
# that each channel does:

ecg_epochs_clean, _ = mne.preprocessing.regress(ecg_epochs, picks_ref='ecg')
fig = ecg_epochs_clean.average(picks=plot_picks).plot(picks=plot_picks)
fig.suptitle('ECG epochs, ECG regressed')
fig.tight_layout()

###############################################################################
# References
# ----------
# .. footbibliography::
