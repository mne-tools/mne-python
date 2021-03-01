"""
====================================================
Compute envelope correlations in volume source space
====================================================

Compute envelope correlations of orthogonalized activity
:footcite:`HippEtAl2012,KhanEtAl2018` in source
space using resting state CTF data in a volume source space.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.connectivity import envelope_correlation
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
bem = op.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')
crop_to = 60.

##############################################################################
# Here we do some things in the name of speed, such as crop (which will
# hurt SNR) and downsample. Then we compute SSP projectors and apply them.

raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, crop_to).pick_types(meg=True, eeg=False).load_data().resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog
raw.apply_proj()
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest

##############################################################################
# Now we band-pass filter our data and create epochs.

raw.filter(14, 30)
events = mne.make_fixed_length_events(raw, duration=5.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                    baseline=None, reject=dict(mag=8e-13), preload=True)
del raw

##############################################################################
# Compute the forward and inverse
# -------------------------------

# This source space is really far too coarse, but we do this for speed
# considerations here
pos = 15.  # 1.5 cm is very broad, done here for speed!
src = mne.setup_volume_source_space('bst_resting', pos, bem=bem,
                                    subjects_dir=subjects_dir, verbose=True)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
data_cov = mne.compute_covariance(epochs)
filters = make_lcmv(epochs.info, fwd, data_cov, 0.05, cov,
                    pick_ori='max-power', weight_norm='nai')
del fwd

##############################################################################
# Compute label time series and do envelope correlation
# -----------------------------------------------------

epochs.apply_hilbert()  # faster to do in sensor space
stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)
corr = envelope_correlation(stcs, verbose=True)

##############################################################################
# Compute the degree and plot it
# ------------------------------

degree = mne.connectivity.degree(corr, 0.15)
stc = mne.VolSourceEstimate(degree, [src[0]['vertno']], 0, 1, 'bst_resting')
brain = stc.plot(
    src, clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
    subjects_dir=subjects_dir, mode='glass_brain')

##############################################################################
# References
# ----------
# .. footbibliography::
