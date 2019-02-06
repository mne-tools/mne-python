"""
=============================================
Compute envelope correlations in source space
=============================================

Compute orthogonal envelope correlations of activity in source space
using resting state CTF data.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.connectivity import envelope_correlation
from mne.datasets import fetch_aparc_sub_parcellation
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
bem = op.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
erm_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_noise_20111104_02.ds')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')

##############################################################################
# Here we do some things in the name of speed, such as crop (which will
# hurt SNR) and downsample. Then we compute SSP projectors and apply them.

raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, 60).load_data().pick_types(meg=True, eeg=False).resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog
raw.apply_proj()
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest

##############################################################################
# Now we band-pass filter our data and create epochs.

raw.filter(14, 30, n_jobs='cuda')
events = mne.make_fixed_length_events(raw, duration=5.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                    baseline=None, reject=None, preload=True)
del raw

##############################################################################
# Compute the forward and inverse

src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
inv = make_inverse_operator(epochs.info, fwd, cov)

##############################################################################
# Compute activity in labels and envelope correlation
fetch_aparc_sub_parcellation(subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub',
                                    subjects_dir=subjects_dir)
labels = mne.morph_labels(labels, subject, subjects_dir=subjects_dir)
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal')
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=True)
corr = envelope_correlation(label_ts)

##############################################################################
# Compute the degree and plot it

degree = mne.connectivity.degree(corr, 0.15)
stc = mne.labels_to_stc(labels, degree)
stc = stc.in_label(mne.Label(src[0]['vertno'], hemi='lh') +
                   mne.Label(src[1]['vertno'], hemi='rh'))
brain = stc.plot(
    clim=dict(kind='percent', lims=[75, 85, 95]),
    subjects_dir=subjects_dir, views='dorsal', hemi='both',
    smoothing_steps=25, time_label='Beta band')
