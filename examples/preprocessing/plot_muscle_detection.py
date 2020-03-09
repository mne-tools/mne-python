"""
===========================
Annotate muscle artifacts
===========================

Muscle contractions produce high frequency activity that can contaminate the
brain signal of interest. Muscle artifacts can be produced when clinching the
jaw, swalloing, or twitching a head muscle. Muscle artifacts are most notable
in teh range of 110-140Hz.

This example uses `annotate_muscle` to annotate segments where muscle activity
likely occurred. This is done by band-pass filtering the data in the 110-140 Hz
range. Then, the envelope is taken to account for peacks and troughs. The
envelope is z-scored and averaged across channels. To remove noisy tansient
peak, the z-scored average is low-pass filtered to 4 Hz. Segments above a set
threshold are annotated as BAD_motion. In addition, `min_length_good` allows to
discard god segments of data between bad segments that are to transient.

Note:
The raw data needs to be notched to remove line activity, otherwise, changes of
AC power could bias the estimation.

The inputted raw data has to be from a single channel type as there might be
different channel type magnitudes.

"""
# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt
from mne import pick_types
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_muscle
from mne.io import concatenate_raws

# Load data

data_path = bst_auditory.data_path()
data_path_MEG = op.join(data_path, 'MEG')
subject = 'bst_auditory'
subjects_dir = op.join(data_path, 'subjects')
trans_fname = op.join(data_path, 'MEG', 'bst_auditory',
                      'bst_auditory-trans.fif')
raw_fname1 = op.join(data_path_MEG, 'bst_auditory', 'S01_AEF_20131218_01.ds')
raw_fname2 = op.join(data_path_MEG, 'bst_auditory', 'S01_AEF_20131218_02.ds')

raw = read_raw_ctf(raw_fname1, preload=False)

raw = concatenate_raws([raw, read_raw_ctf(raw_fname2, preload=False)])
raw.crop(350, 410).load_data()
raw.resample(300, npad="auto")
raw.notch_filter([50, 110, 140])

picks = pick_types(raw.info, meg=True, ref_meg=False)

# detect muscle artifacts
threshold_muscle = 1.5  # z-score
annotation_muscle, scores_muscle = annotate_muscle(raw, picks=picks,
                                                   threshold=threshold_muscle,
                                                   min_length_good=0.2)

###############################################################################
# Plot movement z-scores across recording
# --------------------------------------------------------------------------

plt.figure()
plt.plot(raw.times, scores_muscle)
plt.axhline(y=threshold_muscle, color='r')
plt.show(block=False)
plt.title('High frequency ')
plt.xlabel('time, s.')
plt.ylabel('zscore')

###############################################################################
# Plot raw with annotated muscles
# --------------------------------------------------------------------------

raw.set_annotations(annotation_muscle)
raw.plot(n_channels=100, duration=20)
