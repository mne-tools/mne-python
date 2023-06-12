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
# By default, Eyelink files will output events for occular events (blinks,
# saccades, fixations), and experiment messages. MNE will store these events
# as `mne.Annotations`. If we are only interested in certain event types from
# the Eyelink file, we can select for these using the ``'create_annotations'``
# argument of `mne.io.read_raw_eyelink`. Here, we will only create annotations
# for blinks, and experiment messages.
#
# The info structure tells us we loaded a monocular recording with 2
# ``'eyegaze'``, channels (X/Y), 1 ``'pupil'`` channel, and 1 ``'stim'``
# channel.

from mne import Epochs, find_events
from mne.io import read_raw_eyelink
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration

eyelink_fname = data_path() / "mono_multi-block_multi-DINS.asc"

raw = read_raw_eyelink(eyelink_fname, create_annotations=["blinks", "messages"])
raw.crop(tmin=0, tmax=146)

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
# the recording. Note that you can access the calibration information directly
# as you would a dictionary:

print(f"Eye calibrated: {first_cal['eye']}")
print(f"Calibration average error: {first_cal['avg_error']}")
print(f"Calibration data {repr(first_cal['points'])})")

# %%
# The calibration points are stored as a :class:`numpy.ndarray`. You can access
# the data for a specific calibration point by indexing the array, or you can access a
# specific field for all calibration points by indexing the field name. For example:
print(f"data for the first point only: {first_cal['points'][0]}")
print(f"offset for each calibration point: {first_cal['points']['offset']}")

# %%
# Let's plot the calibration to get a better look. We'll pass
# ``show_offsets=True`` to show the offsets (in visual degrees) between the
# calibration position and the actual gaze position of each calibration point.

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
# the screen. We now extract these events to visualize the pupil response.

events = find_events(raw, "DIN", shortest_event=1, min_duration=0.02, uint_cast=True)
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
# Plot average pupil response
# ---------------------------
#
# We now visualize the pupillary light reflex.
# Therefore, we select only the pupil channel and plot the evoked response to
# the light flashes.
#
# As we see, there is a prominent decrease in pupil size following the
# stimulation. The noise starting about 2.5 s after stimulus onset stems from
# eyeblinks and artifacts in some of the 16 trials.

epochs = Epochs(raw, events, tmin=-0.3, tmax=5, event_id=event_dict, preload=True)
epochs.pick_types(eyetrack="pupil")
epochs.average().plot()

# %%
# It is important to note that pupil size data are reported by Eyelink (and
# stored internally by MNE) as arbitrary units (AU). While it often can be
# preferable to convert pupil size data to millimeters, this requires
# information that is not present in the file. MNE does not currently
# provide methods to convert pupil size data.
# See :ref:`tut-importing-eyetracking-data` for more information on pupil size
# data.
