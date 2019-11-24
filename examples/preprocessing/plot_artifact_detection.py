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
                                                  detect_movement,
                                                  detect_muscle)
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


# Detect bad channels
bad_chns = detect_bad_channels(raw, zscore_v=4, method='both', start=0,
                               end=140, neigh_max_distance=.035)

# detect excecive movement and correct dev_head trans
pos = mne.chpi._calculate_head_pos_ctf(raw)


pos[:, 0] -= raw._first_time  # That is temporary, shouldn't happen


thr_mov = .001  # in meters
annotation_movement, hpi_disp, dev_head_t = detect_movement(raw.info, pos,
                                                            thr_mov=thr_mov)
raw.info['dev_head_t'] = dev_head_t
# Plot movement
plt.figure()
plt.plot(pos[:, 0], hpi_disp)
plt.axhline(y=thr_mov, color='r')
plt.xlabel('time s.')
plt.ylabel('distance m')
plt.title('cHPI w.r.t median recording head position')
plt.show(block=False)

# detect muscle artifacts
thr_mus = 1.5  # z-score
annotation_muscle, scores_muscle = detect_muscle(raw, thr=thr_mus, t_min=0,
                                                 notch=None)

plt.figure()
plt.plot(raw.times, scores_muscle)
plt.axhline(y=thr_mus, color='r')
plt.show(block=False)
plt.title('Avg z-score high freq. activity')
plt.xlabel('time s.')
plt.ylabel('zscore')


raw.set_annotations(annotation_movement + annotation_muscle)

raw.plot(n_channels=100)
