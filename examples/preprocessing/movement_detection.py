# -*- coding: utf-8 -*-
"""
.. _ex-movement-detect:

=====================================================
Annotate movement artifacts and reestimate dev_head_t
=====================================================

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
# License: BSD-3-Clause

# %%

import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_movement, compute_average_dev_head_t

# Load data
data_path = bst_auditory.data_path()
data_path_MEG = data_path / 'MEG'
subject = 'bst_auditory'
subjects_dir = data_path / 'subjects'
trans_fname = data_path / 'MEG' / 'bst_auditory' / 'bst_auditory-trans.fif'
raw_fname1 = data_path_MEG / 'bst_auditory' / 'S01_AEF_20131218_01.ds'
raw_fname2 = data_path_MEG / 'bst_auditory' / 'S01_AEF_20131218_02.ds'
# read and concatenate two files, ignoring device<->head mismatch
raw = read_raw_ctf(raw_fname1, preload=False)
mne.io.concatenate_raws(
    [raw, read_raw_ctf(raw_fname2, preload=False)], on_mismatch='ignore')
raw.crop(350, 410).load_data()
raw.resample(100, npad="auto")

# %%
# Plot continuous head position with respect to the mean recording position
# --------------------------------------------------------------------------

# Get cHPI time series and compute average
chpi_locs = mne.chpi.extract_chpi_locs_ctf(raw)
head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
original_head_dev_t = mne.transforms.invert_transform(
    raw.info['dev_head_t'])
average_head_dev_t = mne.transforms.invert_transform(
    compute_average_dev_head_t(raw, head_pos))
fig = mne.viz.plot_head_positions(head_pos)
for ax, val, val_ori in zip(fig.axes[::2], average_head_dev_t['trans'][:3, 3],
                            original_head_dev_t['trans'][:3, 3]):
    ax.axhline(1000 * val, color='r')
    ax.axhline(1000 * val_ori, color='g')

# The green horizontal lines represent the original head position, whereas the
# red lines are the new head position averaged over all the time points.

# %%
# Plot raw data with annotated movement
# ------------------------------------------------------------------

mean_distance_limit = .0015  # in meters
annotation_movement, hpi_disp = annotate_movement(
    raw, head_pos, mean_distance_limit=mean_distance_limit)
raw.set_annotations(annotation_movement)
raw.plot(n_channels=100, duration=20)

##############################################################################
# After checking the annotated movement artifacts, calculate the new transform
# and plot it:
new_dev_head_t = compute_average_dev_head_t(raw, head_pos)
raw.info['dev_head_t'] = new_dev_head_t
fig = mne.viz.plot_alignment(raw.info, show_axes=True, subject=subject,
                             trans=trans_fname, subjects_dir=subjects_dir)
mne.viz.set_3d_view(fig, azimuth=90, elevation=60)
