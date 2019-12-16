"""
=============================================
Detect artifacts
=============================================

Detects bad channels

Annotates periods where there is excessive movement and calculates a new
device to head transformation matrix

Annotates segments contaminated with muscle artifacts

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)

import os.path as op
import mne

from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing.artifact_detection import (detect_bad_channels,
                                                  annotate_movement,
                                                  annotate_muscle,
                                                  compute_average_dev_head_t)
import matplotlib.pyplot as plt

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
raw.filter(l_freq=1, h_freq=149, picks='meg')


# Detect bad channels
bad_chns = detect_bad_channels(raw, zscore_v=4, method='both', tmin=0,
                               tmax=140, neigh_max_distance=.035)

# detect excecive movement and correct dev_head trans
pos = mne.chpi._calculate_head_pos_ctf(raw)

thr_mov = .0015  # in meters
out = annotate_movement(raw, pos, displacement_limit=thr_mov)
annotation_movement, hpi_disp = out

# Plot movement
raw.set_annotations(annotation_movement)
raw.plot(n_channels=100, duration=20)

plt.figure()
plt.plot(pos[:, 0], hpi_disp)
plt.axhline(y=thr_mov, color='r')
plt.xlabel('time s.')
plt.ylabel('distance m')
plt.title('cHPI w.r.t median recording head position')
plt.show(block=False)

# detect muscle artifacts
thr_mus = 1.5  # z-score
annotation_muscle, scores_muscle = annotate_muscle(raw, thr=thr_mus, t_min=0,
                                                   notch=None)

# Plot muscle
plt.figure()
plt.plot(raw.times, scores_muscle)
plt.axhline(y=thr_mus, color='r')
plt.show(block=False)
plt.title('Avg z-score high freq. activity')
plt.xlabel('time s.')
plt.ylabel('zscore')

raw.set_annotations(annotation_muscle)
raw.plot(n_channels=100, duration=20)

# Change dev to head transform
new_dev_head_t = compute_average_dev_head_t(raw, pos)
raw.info['dev_head_t'] = new_dev_head_t



