# -*- coding: utf-8 -*-
"""
.. _tut-eyetrack:

===========================================
Working with eye tracker data in MNE-Python
===========================================

In this tutorial we will load some eye tracker data and plot the average
pupil response to light flashes (i.e. the pupillary light reflex).

"""  # noqa: E501
# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <scott.huberty@mail.mcgill.ca>
#
# License: BSD-3-Clause

# %%
# Data loading
# ------------
#
# First we will load an eye tracker recording from SR research's proprietary
# ``'.asc'`` file format.
#
# The info structure tells us we loaded a monocular recording with 2
# ``'eyegaze'``, channels (X/Y), 1 ``'pupil'`` channel, and 1 ``'stim'``
# channel.

import mne
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration

eyelink_fname = data_path() / "mono_multi-block_multi-DINS.asc"

raw = mne.io.read_raw_eyelink(eyelink_fname, create_annotations=["blinks", "messages"])
raw.crop(tmin=0, tmax=130)  # for this demonstration, let's take a subset of the data

# %%
# Ocular annotations
# ------------------
# By default, Eyelink files will output events for ocular events (blinks,
# saccades, fixations), and experiment messages. MNE will store these events
# as `mne.Annotations`. Ocular annotations contain channel information, in the
# ``'ch_names'``` key. This means that we can see which eye an ocular event occurred in:

print(raw.annotations[0])  # a blink in the right eye

# %%
# If we are only interested in certain event types from
# the Eyelink file, we can select for these using the ``'create_annotations'``
# argument of `mne.io.read_raw_eyelink`. above, we only created annotations
# for blinks, and experiment messages.
#
# Note that ``'blink'`` annotations are read in as ``'BAD_blink'``, and MNE will treat
# these as bad segments of data. This means that blink periods will be dropped during
# epoching by default.

# %%
# Checking the calibration
# ------------------------
#
# We can also load the calibrations from the recording and visualize them.
# Checking the quality of the calibration is a useful first step in assessing
# the quality of the eye tracking data. Note that
# :func:`~mne.preprocessing.eyetracking.read_eyelink_calibration`
# will return a list of :class:`~mne.preprocessing.eyetracking.Calibration` instances,
# one for each calibration. We can index that list to access a specific calibration.

cals = read_eyelink_calibration(eyelink_fname)
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
# Get stimulus events from DIN channel
# ------------------------------------
#
# Eyelink eye trackers have a DIN port that can be used to feed in stimulus
# or response timings. :func:`mne.io.read_raw_eyelink` loads this data as a
# ``'stim'`` channel. Alternatively, the onset of stimulus events could be sent
# to the eyetracker as ``messages`` - these can be read in as
# `mne.Annotations`.
#
# In the example data, the DIN channel contains the onset of light flashes on
# the screen. We now extract these events to visualize the pupil response. We will use
# these later in this tutorial.

events = mne.find_events(
    raw, "DIN", shortest_event=1, min_duration=0.02, uint_cast=True
)
event_dict = {"flash": 3}


# %%
# Plot raw data
# -------------
#
# As the following plot shows, we now have a raw object with the eye tracker
# data, eyeblink annotations and stimulus events (from the DIN channel).
#
# The plot also shows us that there is some noise in the data (not always
# categorized as blinks). Also, notice that we have passed a custom `dict` into
# the scalings argument of ``raw.plot``. This is necessary to make the eyegaze
# channel traces legible when plotting, since the file contains pixel position
# data (as opposed to eye angles, which are reported in radians). We also could
# have simply passed ``scalings='auto'``.

raw.plot(
    events=events,
    event_id={"Flash": 3},
    event_color="g",
    start=25,
    duration=45,
    scalings=dict(eyegaze=1e3),
)

# %%
# Dealing with artifacts
# ----------------------
# From the plot above, we see that there are some artifacts in the data that we should
# remove before analyzing the pupil response. First, we notice that there is some
# high frequency noise in the pupil signal, likely due to the sub-optimal calibration
# of the eye tracker. We can remove this noise by low-pass filtering the data:

# Apply a low pass filter to the pupil channel
raw.filter(l_freq=None, h_freq=40, picks=["pupil_right"])

# %%
# Dealing with blink artifacts
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We also notice that, naturally, there are blinks in our data, and these blink periods
# occur within ``"blink_R"``  annotations. During blink periods, ``"eyegaze"``
# coordinates are not reported, and ``"pupil"`` size data are ``0``. We don't want these
# blink artifacts biasing our analysis, so we have two options: We can either remove the
# blink periods from our data (by using ``raw.annotations.rename({blink_R: bad_blink})``
# so that the blinks are removed before epoching), or we can interpolate the pupil size
# data during the blink periods. If we were interested in analyzing the eye position
# data, we would need to remove the blink periods from our data. However, since we are
# only interested in the pupil size data, let's interpolate the missing pupil sizes in
# the ``"pupil"`` channel during blinks:

# TODO: remove comment after PR 11746 is merged
# mne.preprocessing.eyetracking.interpolate_blinks(raw, buffer=0.05)
# Let's plot our data again to see the result of the interpolation:
raw.pick(["pupil_right"])  # Let's pick just the pupil channel
raw.plot(events=events, event_id={"Flash": 3}, event_color="g")

# %%
# :func:`~mne.preprocessing.eyetracking.interpolate_blinks` performs a simple linear
# interpolation of the pupil size data during blink periods. the ``buffer`` keyword
# argument specifies the amount of time (in seconds) before and after the blinks to
# include in the interpolation. This is helpful because the ``blink`` annotations
# do not always capture the entire blink in the signal. We specified a value of ``.05``
# seconds (50 ms), which is slightly more than the default value of ``.025``.

# %%
# Rejecting bad spans of data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Even after filtering the pupil data and interpolating the blink periods, we still see
# some artifacts in the data (the large spikes) that we don't want to include in our
# analysis. Let's epoch our data and then reject any epochs that might contain these
# artifacts. We'll use :class:`mne.Epochs` to epoch our data, and pass in the
# ``events`` array and ``event_dict`` that we created earlier. We'll also pass in the
# ``reject`` keyword argument to reject any epochs that contain data that exceeds a
# peak-to-peak signal amplitude threshold of ``1500`` in the ``"pupil"`` channel.
# Note that this threshold is arbitrary, and should be adjusted based on the data.
# We chose 1500 because eyelink reports pupil size in arbitrary units (AU), which
# typically ranges from 800 to 3000 units. Our epochs already contains large
# signal fluctuations due to the pupil response, so a threshold of 1500 is conservative
# enough to reject epochs only with large artifacts.

epochs = mne.Epochs(
    raw,
    events,
    tmin=-0.3,
    tmax=5,
    event_id=event_dict,
    preload=True,
    reject=dict(pupil=1500),
)
epochs.plot()

# %%
# We can clearly see the prominent decrease in pupil size following the
# stimulation.

# %%
# Plot average pupil response
# ---------------------------
#
# Finally, let's plot the evoked response to the light flashes to get a sense of the
# average pupillary light response.

epochs.average().plot()

# %%
# Again, it is important to note that pupil size data are reported by Eyelink (and
# stored internally by MNE) as arbitrary units (AU). While it often can be
# preferable to convert pupil size data to millimeters, this requires
# information that is not present in the file. MNE does not currently
# provide methods to convert pupil size data.
# See :ref:`tut-importing-eyetracking-data` for more information on pupil size
# data.
