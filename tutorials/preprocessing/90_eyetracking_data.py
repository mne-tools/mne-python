# -*- coding: utf-8 -*-
"""
.. _tut-eyetrack:

===========================================
Working with eye tracker data in MNE-Python
===========================================

In this tutorial we will explore simultaneously recorded eye-tracking and EEG data from
a pupillary light reflex task. We will combine the eye-tracking and EEG data, and plot
the ERP and pupil response to the light flashes (i.e. the pupillary light reflex).

"""  # noqa: E501
# Authors: Scott Huberty <seh33@uw.edu>
#          Dominik Welke <dominik.welke@web.de>
#
#
# License: BSD-3-Clause

# %%
# Data loading
# ------------
#
# As usual we start by importing the modules we need and loading some
# :ref:`example data <eyelink-dataset>`: eye-tracking data recorded from SR research's
# ``'.asc'`` file format, and EEG data recorded from EGI's ``'.mff'`` file format.
#
# The info structure of the eye-tracking data tells us we loaded a monocular recording
# with 2 ``'eyegaze'``, channels (X/Y), 1 ``'pupil'`` channel, 1 ``'stim'`` channel, and
# 3 channels for the head distance and position (since this data was collected using
# EyeLink's Remote mode).

import mne
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration

et_fpath = data_path() / "sub-01_task-plr_eyetrack.asc"
eeg_fpath = data_path() / "sub-01_task-plr_eeg.mff"

raw_et = mne.io.read_raw_eyelink(et_fpath, create_annotations=["blinks"])
raw_eeg = mne.io.read_raw_egi(eeg_fpath, preload=True, verbose="warning")
raw_eeg.filter(1, 30, verbose="warning")
print(f"RawEyelink info: {raw_et.info}")

# %%
# .. seealso:: :ref:`tut-importing-eyetracking-data`
     :class: sidebar

# %%
# Ocular annotations
# ------------------
# By default, EyeLink files will output events for ocular events (blinks,
# saccades, fixations), and experiment messages. MNE will store these events
# as `mne.Annotations`. Ocular annotations contain channel information, in the
# ``'ch_names'`` key. This means that we can see which eye an ocular event occurred in,
# which can be useful for binocular recordings:

print(raw_et.annotations[0]["ch_names"]) # a blink in the right eye

# %%
# If we are only interested in certain event types from
# the EyeLink file, we can select for these using the ``create_annotations``
# argument of `~mne.io.read_raw_eyelink`. Above, we only created annotations
# for blinks, which are read in as ``'BAD_blink'`` so that MNE will treat
# these as bad segments of data.

# %%
# Checking the calibration
# ------------------------
#
# EyeLink ``.asc`` files can also include calibration information.
# MNE-Python can load and visualize those eye-tracking calibrations, which
# is a useful first step in assessing the quality of the eye-tracking data.
# :func:`~mne.preprocessing.eyetracking.read_eyelink_calibration`
# will return a list of :class:`~mne.preprocessing.eyetracking.Calibration` instances,
# one for each calibration. We can index that list to access a specific calibration.

cals = read_eyelink_calibration(et_fpath)
print(f"number of calibrations: {len(cals)}")
first_cal = cals[0]  # let's access the first (and only in this case) calibration
print(first_cal)

# %%
# Here we can see that a 5-point calibration was performed at the beginning of
# the recording. Note that you can access the calibration information using
# dictionary style indexing:

print(f"Eye calibrated: {first_cal['eye']}")
print(f"Calibration model: {first_cal['model']}")
print(f"Calibration average error: {first_cal['avg_error']}")

# %%
# The data for individual calibration points are stored as :class:`numpy.ndarray`
# arrays, in the ``'positions'``, ``'gaze'``, and ``'offsets'`` keys. ``'positions'``
# contains the x and y coordinates of each calibration point. ``'gaze'`` contains the
# x and y coordinates of the actual gaze position for each calibration point.
# ``'offsets'`` contains the offset (in visual degrees) between the calibration position
# and the actual gaze position for each calibration point. Below is an example of
# how to access these data:
print(f"offset of the first calibration point: {first_cal['offsets'][0]}")
print(f"offset for each calibration point: {first_cal['offsets']}")
print(f"x-coordinate for each calibration point: {first_cal['positions'].T[0]}")

# %%
# Let's plot the calibration to get a better look. Below we see the location that each
# calibration point was displayed (gray dots), the positions of the actual gaze (red),
# and the offsets (in visual degrees) between the calibration position and the actual
# gaze position of each calibration point.

first_cal.plot(show_offsets=True)

# %%
# Extract common stimulus events from the data
# --------------------------------------------
#
# In this experiment, a photodiode attached to the display screen was connected to both
# the EEG and eye-tracking systems. The photodiode was triggered by the the light flash
# stimuli, causing a signal to be sent to both systems simultaneously, signifying the
# onset of the flash. The photodiode signal was recorded as a digital input channel in
# the EEG and eye-tracking data. MNE loads these data as ``'stim'`` channels.
#
# We'll extract the flash event onsets from both the EEG and eye-tracking data, as they
# are necessary for aligning the EEG and eye-tracking data.

et_events = mne.find_events(raw_et, min_duration=0.01, shortest_event=1, uint_cast=True)
event_dict = {"Flash": 2}
eeg_events = mne.find_events(raw_eeg, stim_channel="DIN3")

# %%
# Plot the raw eye-tracking data
# ------------------------------
#
# Let's plot the raw eye-tracking data. We'll pass a custom `dict` into
# the scalings argument to make the eyegaze channel traces legible when plotting,
# since this file contains pixel position data (as opposed to eye angles,
# which are reported in radians). We also could have simply passed ``scalings='auto'``.

raw_et.plot(
    events=et_events,
    event_id=event_dict,
    event_color="g",
    duration=15,
    scalings=dict(eyegaze=1e3),
)

# %%
# Handling blink artifacts
# ------------------------
#
# Naturally, there are blinks in our data, and these blink periods
# occur within ``"BAD_blink"``  annotations. During blink periods, ``"eyegaze"``
# coordinates are not reported, and ``"pupil"`` size data are ``0``. We don't want these
# blink artifacts biasing our analysis, so we have two options: Drop the blink periods
# from our data during epoching, or interpolate the missing data during the blink
# periods. For this tutorial, let's interpolate the blink samples:

mne.preprocessing.eyetracking.interpolate_blinks(raw_et, buffer=(0.05, 0.2))

# %%
# .. important:: By default, :func:`~mne.preprocessing.eyetracking.interpolate_blinks`, will
#           only interpolate blinks in ``"pupil"`` channels. Passing
#           ``interpolate_gaze=True`` will also interpolate the blink periods of the
#           ``"eyegaze"`` channels. Be aware, however, that eye movements can occur
#           during blinks which makes the gaze data less suitable for interpolation.

# %%
# Aligning the eye-tracking data with EEG data
# --------------------------------------------
#
# In this dataset, eye-tracking and EEG data were recorded simultaneously, but on different
# systems, so we'll need to align the data before we can analyze them together. We can
# do this using the :func:`~mne.preprocessing.realign_raw` function, which will align
# the data based on the timing of the shared events that are present in both
# :class:`~mne.io.Raw` objects. We'll use the shared photodiode events we extracted
# above, but first we need to convert the event onsets from samples to seconds. Once the
# data have been aligned, we'll add the EEG channels to the eye-tracking raw object.

# Convert event onsets from samples to seconds
et_din_times = et_events[:, 0] / raw_et.info["sfreq"]
eeg_din_times = eeg_events[:, 0] / raw_eeg.info["sfreq"]
# Align the data
mne.preprocessing.realign_raw(
    raw_et, raw_eeg, et_din_times, eeg_din_times, verbose="error"
)
# Add EEG channels to the eye-tracking raw object
raw_et.add_channels([raw_eeg], force_update_info=True)

# Define a few channel groups of interest and plot the data
frontal = ["E19", "E11", "E4", "E12", "E5"]
occipital = ["E61", "E62", "E78", "E67", "E72", "E77"]
pupil = ["pupil_right"]
picks_idx = mne.pick_channels(
    raw_et.ch_names, frontal + occipital + pupil, ordered=True
)
raw_et.plot(events=et_events, event_id=event_dict, event_color="g", order=picks_idx)


# %%
# Epoching the data
# -----------------

epochs = mne.Epochs(
    raw_et,
    events=et_events,
    event_id=event_dict,
    tmin=-0.3,
    tmax=3,
    preload=True,
)
epochs[:8].plot(events=et_events, event_id=event_dict, order=picks_idx)

# %%
# We can clearly see the prominent decrease in pupil size following the
# stimulation.

# %%
# Plot the evoked response
# ------------------------
#
# Finally, let's plot the evoked responses to the light flashes to get a sense of the
# average pupillary light response, and the ERP in the EEG data.

epochs.average().plot(picks=occipital + pupil)
