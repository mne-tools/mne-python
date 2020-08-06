# -*- coding: utf-8 -*-
"""
.. _tut-artifact-regression:

Repairing artifacts with regression
===================================

This tutorial covers removal of artifacts using regression as in
:footcite:`GrattonEtAl1983`.

.. contents:: Page contents
   :local:
   :depth: 2

We begin as always by importing the necessary Python modules and loading some
data. In this case we use data from :ref:`brainstorm <tut-brainstorm-auditory>`
because it has both EOG and ECG electrodes. We then crop it to 60 seconds,
set some channel information, then process the data.

Note that there are other corrections that are useful for this dataset that we
will not apply here — see :ref:`tut-brainstorm-auditory` for more information.
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
# Generally speaking, artifacts whose time waveforms on the sensors are
# accurately reflected by some reference signal can be removed by regression.
# Two examples of this are blink artifacts (captured by EOG channels), and
# heartbeat artifacts (captured by ECG electrodes).
#
#
# Example: EOG artifacts
# ~~~~~~~~~~~~~~~~~~~~~~
#
# First let's epoch our data the normal way, using just event type 1 (only the
# "standard" trials) for simplicity. Event timing is adjusted by comparing the
# trigger times on detected sound onsets on channel UADC001-4408.

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
# Next we'll compare the `~mne.Evoked` data (average across epochs) before and
# after we regress out the EOG signal. We do this by estimating the regression
# coefficients on epoch data with the evoked response subtracted out, then use
# those coefficients to remove the EOG signal from the original data:

# do regression
_, betas = mne.preprocessing.regress_artifact(epochs.copy().subtract_evoked())
epochs_clean, _ = mne.preprocessing.regress_artifact(epochs, betas=betas)

# get ready to plot
plot_picks = ['meg', 'eog', 'ecg']
evo_kwargs = dict(picks=plot_picks, spatial_colors=True,
                  verbose='error')  # ignore warnings about spatial colors

# plot original data (averaged across epochs)
fig = epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('Auditory epochs')
fig.tight_layout()

# plot regressed data
fig = epochs_clean.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('Auditory epochs, EOG regressed')
fig.tight_layout()

# clean up
del epochs, epochs_clean

###############################################################################
# The effect is subtle in these evoked data, but you can see that a bump toward
# the end of the window has had its amplitude decreased.
#
# The effect is clearer if we create epochs around (autodetected) blink events,
# and plot the average blink artifact before and after the regression (note
# that the y-axis limits change between the left and right plots):

# extract epochs around each blink
eog_epochs = mne.preprocessing.create_eog_epochs(raw, decim=decim)
eog_epochs.apply_baseline((None, None))

# regress, using the `betas` we already found above
eog_epochs_clean, _ = mne.preprocessing.regress_artifact(eog_epochs,
                                                         betas=betas)

# plot original blink epochs
fig = eog_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('EOG epochs')
fig.tight_layout()

# plot regressed blink epochs
fig = eog_epochs_clean.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('EOG epochs, EOG regressed')
fig.tight_layout()

###############################################################################
# We can also apply the regression directly to the raw data. To do this relies
# on first computing the regression weights *from epoched data with the evoked
# response subtracted out* (as we did above).  If instead one computed
# regression weights from the raw data, it is likely that some brain signal
# would also get removed.

# get ready to plot
order = np.concatenate([  # plotting order: EOG+ECG first, then MEG
    mne.pick_types(raw.info, meg=False, eog=True, ecg=True),
    mne.pick_types(raw.info, meg=True)])
raw_kwargs = dict(events=eog_epochs.events, order=order, start=20, duration=5,
                  n_channels=40)

# plot original data
raw.plot(**raw_kwargs)

# regress (using betas computed above) & plot
raw_clean, _ = mne.preprocessing.regress_artifact(raw, betas=betas)
raw_clean.plot(**raw_kwargs)

del eog_epochs, eog_epochs_clean, raw_clean  # save memory

###############################################################################
# Example: ECG artifacts
# ~~~~~~~~~~~~~~~~~~~~~~
# We can follow the same process with ECG, although it turns out not to work as
# well. This is likely because the ECG signal is like a rotating dipole, and
# therefore the ECG electrode's time waveform does not reflect the same
# temporal dynamics that manifest at each MEG channel. Other approaches like
# :ref:`ICA <tut-artifact-ica>` or :ref:`SSP <tut-artifact-ssp>` may work
# better with this dataset.

# extract epochs around each heartbeat
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, decim=decim)
ecg_epochs.apply_baseline((None, None))

# plot original heartbeat epochs
fig = ecg_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('ECG epochs')
fig.tight_layout()

# regress (here we operate in place to save memory)
mne.preprocessing.regress_artifact(
    ecg_epochs, picks_artifact='ecg', copy=False)

# plot regressed heartbeat epochs
fig = ecg_epochs.average(picks=plot_picks).plot(**evo_kwargs)
fig.suptitle('ECG epochs, ECG regressed')
fig.tight_layout()

###############################################################################
# References
# ^^^^^^^^^^
# .. footbibliography::
