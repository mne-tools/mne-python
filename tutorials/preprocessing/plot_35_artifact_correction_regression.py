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
raw.pick_types(meg=True, stim=True, misc=True,
               eog=True, ecg=True, ref_meg=False).load_data()
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']
raw.filter(None, 40)
decim = 12  # 2400 -> 200 Hz sample rate for epochs

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
epochs = mne.Epochs(raw, events, event_id=1, decim=decim, preload=True)

###############################################################################
# Now let's regress our EOG signal and plot the original and processed data.
# We do this by estimating the regression coefficients on data without the
# evoked response, then use those coefficients to remove the EOG signal:

plot_picks = ['meg', 'eog', 'ecg']
evo_kwargs = dict(picks=plot_picks, spatial_colors=True,
                  verbose='error')  # ignore warnings about spatial colors
fig = epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('Auditory epochs')
fig.tight_layout()

epochs_no_ave = epochs.copy().subtract_evoked()
_, betas = mne.preprocessing.regress(epochs_no_ave)
epochs_clean, _ = mne.preprocessing.regress(epochs, betas=betas)
fig = epochs_clean.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('Auditory epochs, EOG regressed')
fig.tight_layout()
del epochs, epochs_clean

###############################################################################
# The effect is subtle in these evoked data. You can see that a bump toward
# the end of the window has had its amplitude decreased.
#
# The effect is even clearer if we create epochs around our (autodetected) EOG
# events and plot the average across those epochs:

eog_epochs = mne.preprocessing.create_eog_epochs(raw, decim=decim)
eog_epochs.apply_baseline((None, None))
order = np.concatenate([  # plotting order: EOG+ECG first, then MEG
    mne.pick_types(raw.info, meg=False, eog=True, ecg=True),
    mne.pick_types(raw.info, meg=True)])
raw_kwargs = dict(order=order, duration=25, n_channels=40)
raw.plot(events=eog_epochs.events, **raw_kwargs)
fig = eog_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('EOG epochs')
fig.tight_layout()

###############################################################################
# And then clean those data:

raw_clean, _ = mne.preprocessing.regress(raw, betas=betas)
raw_clean.plot(events=eog_epochs.events, **raw_kwargs)
eog_epochs_clean, _ = mne.preprocessing.regress(eog_epochs, betas=betas)
fig = eog_epochs_clean.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('EOG epochs, EOG regressed')
fig.tight_layout()
del eog_epochs, eog_epochs_clean, raw_clean  # save memory

###############################################################################
# ECG
# ~~~
# We can follow the same process with ECG:

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, decim=decim)
ecg_epochs.apply_baseline((None, None))
raw.plot(events=ecg_epochs.events, **raw_kwargs)
fig = ecg_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('ECG epochs')
fig.tight_layout()

###############################################################################
# But it does not remove as much of the signal, even from the reference
# epochs, likely because the ECG signal is a rotating dipole, and the ECG
# electrode time waveform does not capture the same effective time waveform
# that each channel does:

# Here we operate in place to save memory
mne.preprocessing.regress(ecg_epochs, picks_ref='ecg', copy=False)
fig = ecg_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('ECG epochs, ECG regressed')
fig.tight_layout()

###############################################################################
# References
# ----------
# .. footbibliography::
