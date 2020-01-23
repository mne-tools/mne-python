"""
======================================================
Annotate movement artifacts and reestimate dev_head_t
======================================================

Periods, where the participant moved considerably, are contaminated by low
amplitude artifacts. When averaging the magnetic fields, the more spread the
head position, the bigger the cancellation due to different locations.
Similarly, the covariance will also be affected by severe head movement,
and source estimation will suffer low/smeared coregistration accuracy.

This example uses the continuous head position indicators (cHPI) times series
to annotate periods of head movement, then the device to head transformation
matrix is estimated from the artifact-free segments. The new head position will
be more representative of the actual head position during the recording.

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt
import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing.artifact_detection import (annotate_movement,
                                                  compute_average_dev_head_t)


# Load data
data_path = bst_auditory.data_path()
data_path_MEG = op.join(data_path, 'MEG')

subject = 'bst_auditory'
subjects_dir = op.join(data_path, 'subjects')

raw_fname1 = op.join(data_path_MEG, 'bst_auditory', 'S01_AEF_20131218_01.ds')
raw_fname2 = op.join(data_path_MEG, 'bst_auditory', 'S01_AEF_20131218_02.ds')

raw = read_raw_ctf(raw_fname1, preload=False)

mne.io.concatenate_raws([raw, read_raw_ctf(raw_fname2, preload=False)])
raw.crop(350, 500).load_data()
raw.resample(300, npad="auto").notch_filter([60, 120])

# get cHPI time series
pos = mne.chpi.calculate_head_pos_ctf(raw)

mean_distance_limit = .0015  # in meters
out = annotate_movement(raw, pos, mean_distance_limit=mean_distance_limit)
annotation_movement, hpi_disp = out

###############################################################################
# Plot continuous head position with respect to the mean recording position
# --------------------------------------------------------------------------
plt.figure()
plt.plot(pos[:, 0], hpi_disp)
plt.axhline(y=mean_distance_limit, color='r')
plt.xlabel('time s')
plt.ylabel('distance m')
plt.title('cHPI w.r.t mean recording head position ')
plt.show(block=False)

###############################################################################
# Plot raw data with annotated movement
# ------------------------------------------------------------------
raw.set_annotations(annotation_movement)
raw.plot(n_channels=100, duration=20)


# After checking the annotated movement artifacts, calculate the new transform
new_dev_head_t = compute_average_dev_head_t(raw, pos)
raw.info['dev_head_t'] = new_dev_head_t
