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
# .asc file format.
# We can select which annotations to read out (the files usually also include
# annotations for saccades and fixations).
#
# As the info structure tells us we loaded a monocular recording with 2 Gaze
# channels (X/Y), 1 Pupil channel, and 1 Stim channel.

from mne import Epochs, find_events
from mne.io import read_raw_eyelink
from mne.datasets.eyelink import data_path

eyelink_fname = data_path() / 'mono_multi-block_multi-DINS.asc'

raw = read_raw_eyelink(eyelink_fname,
                       create_annotations=['blinks', 'messages'])
raw.crop(tmin=0, tmax=146)

# %%
# Get stimulus events from DIN channel
# ------------------------------------
#
# Eyelink eye trackers have a DIN port that can be used to feed in stimulus
# or response timings. :func:`mne.io.read_raw_eyelink` loads this data as a
# Stim channel.
# Alternatively, trigger information could be send to the eyetracker as
# `messages` - these can be read in as annotations.
#
# In the example data, the DIN channel contains the onset of light flashes on
# screen. We now extract these events to visualize the pupil response.

events = find_events(raw, 'DIN',
                     shortest_event=1,
                     min_duration=.02,
                     uint_cast=True)
event_dict = {'flash': 3}


# %%
# Plot raw data
# -------------
#
# As the following plot shows, we now have a raw object with the eye tracker
# data, eyeblink annotations and stimulus events (from the DIN channel).
#
# The plot also shows us that there is some noise in the data (not always
# categorized as blinks).

raw.plot(events=events, event_id={'Flash': 3}, event_color='g',
         start=25, duration=45)


# %%
# Plot average pupil response
# ---------------------------
#
# We now visualize the pupillary light reflex.
# Therefor, we select only the pupil channel and plot the evoked response to
# the light flashes.
#
# As we see, there is a prominent decrease in pupil size following the
# stimulation. The noise starting about 2.5 s after stimulus onset stems from
# eyeblinks and artifacts in some of the 16 trials.

epochs = Epochs(raw, events, tmin=-0.3, tmax=5,
                event_id=event_dict, preload=True)
epochs.pick_types(eyetrack='eyetrack_pupil')
epochs.average().plot()

# %%
