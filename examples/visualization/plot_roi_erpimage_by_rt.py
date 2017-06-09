"""
========================================================================
Plot single trial activity, EEG channels grouped by ROI and sorted by RT
========================================================================

This will produce what is sometimes called an event related
potential / field (ERP/ERF) image.

The EEGLAB example file is read in and response times are calculated.
ROIs are determined by the channel types (in 10/20 channel notation,
even channels are right, odd are left, and 'z' are central). The
median within each channel is calculated, and the trials are plotted,
sorted by response time.
"""
# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD (3-clause)

from numpy import median
import matplotlib.pyplot as plt

import mne
from mne.datasets import testing
from mne import Epochs, io, pick_types
from mne.event import define_target_events

print(__doc__)

###############################################################################
# Load EEGLAB example data (a small EEG dataset)
data_path = testing.data_path()
fname = data_path + "/EEGLAB/test_raw.set"
montage = data_path + "/EEGLAB/test_chans.locs"

event_id = {"rt": 1, "square": 2}  # must be specified for str events
eog = {"FPz", "EOG1", "EOG2"}
raw = io.eeglab.read_raw_eeglab(
    fname, preload=True, eog=eog, montage=montage, event_id=event_id)
picks = pick_types(raw.info, eeg=True)
events = mne.find_events(raw)

###############################################################################
# Create Epochs

# define target events:
# 1. find response times: distance between "square" and "rt" events
# 2. extract A. "square" events B. followed by a button press within 700 msec
tmax = .7
events, rts = define_target_events(
    events, 2, 1, raw.info["sfreq"], 0., tmax, 2)

epochs = Epochs(raw, events=events, tmax=tmax + .1,
                event_id={"square": 2}, picks=picks)

###############################################################################
# construct ROIs
rois = dict()
for pick, channel in enumerate(epochs.ch_names):
    last_char = channel[-1]
    if last_char == "z":  # midline
        roi = "Midline"
        rois[roi] = rois.get(roi, list()) + [pick]
    else:
        last_char = int(last_char)
        roi = "Left" if last_char % 2 else "Right"
        rois[roi] = rois.get(roi, list()) + [pick]

# set up corresponding axes to plot to
axes = dict()
for ii, roi in enumerate(sorted(rois.keys())):
    im_ax = plt.subplot2grid((3, 3), (0, ii), colspan=1, rowspan=2)
    ts_ax = plt.subplot2grid((3, 3), (2, ii), colspan=1, rowspan=1)
    axes[roi] = [im_ax, ts_ax]

###############################################################################
# Plot
overlay_times = rts / 1000  # RT in seconds
order = rts.argsort()  # sorting from fast to slow trials
combine = lambda data: median(data, 1)  # take the median of each ROI

epochs["square"].plot_image(
    groupby=rois, combine=combine, axes=axes,
    overlay_times=overlay_times, order=order, colorbar=False)
for roi in ("Midline", "Right"):
    for ax in axes[roi]:
        ax.set_ylabel('')
        ax.set_yticks(())
plt.show()