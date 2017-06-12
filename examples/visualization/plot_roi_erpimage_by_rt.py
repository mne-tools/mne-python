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
raw = io.eeglab.read_raw_eeglab(fname, preload=True, eog=eog,
                                montage=montage, event_id=event_id)
picks = pick_types(raw.info, eeg=True)
events = mne.find_events(raw)

###############################################################################
# Create Epochs

# define target events:
# 1. find response times: distance between "square" and "rt" events
# 2. extract A. "square" events B. followed by a button press within 700 msec
tmax = .7
sfreq = raw.info["sfreq"]
reference_id, target_id = 2, 1
new_events, rts = define_target_events(events, reference_id, target_id, sfreq,
                                       tmin=0., tmax=tmax, new_id=2)

epochs = Epochs(raw, events=new_events, tmax=tmax + .1,
                event_id={"square": 2}, picks=picks)

###############################################################################
# Plot

## construct ROIs
rois = dict()
for pick, channel in enumerate(epochs.ch_names):
    last_char = channel[-1]
    if last_char == "z":
        roi = "Midline"
    else:
        roi = "Left" if int(last_char) % 2 else "Right"
    rois[roi] = rois.get(roi, list()) + [pick]

## set up corresponding axes to plot to
axes = dict()
for ii, roi in enumerate(sorted(rois.keys())):
    im_ax = plt.subplot2grid((3, 3), (0, ii), colspan=1, rowspan=2)
    ts_ax = plt.subplot2grid((3, 3), (2, ii), colspan=1, rowspan=1)
    axes[roi] = [im_ax, ts_ax]

## Parameters for plotting
overlay_times = rts / 1000  # RT in seconds
order = rts.argsort()  # sorting from fast to slow trials


def combine(data):  # we will take the median of each ROI
    return median(data, axis=1)

## The actual plot
epochs["square"].plot_image(
    groupby=rois, combine=combine, axes=axes,
    overlay_times=overlay_times, order=order, colorbar=False)
for roi in ("Midline", "Right"):  # uncrowd visually
    for ax in axes[roi]:
        ax.set_ylabel('')
        ax.set_yticks(())
plt.show()
