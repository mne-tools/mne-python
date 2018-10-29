"""
================================
Brainstorm resting state dataset
================================

Here we compute the resting state from raw for the
Brainstorm tutorial dataset, see [1]_.

The pipeline is meant to mirror the Brainstorm
`resting tutorial pipeline <bst_tut_>`_. The steps we use are:

1. Filtering: downsample heavily.
2. Artifact detection: use SSP for EOG and ECG.
3. Source localization: dSPM, depth weighting, cortically constrained.
4. Frequency: power spectrum density (Welch), 4 sec window, 50% overlap.
5. Standardize: normalize by relative power for each source.

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716

.. _bst_tut: https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega
"""
# sphinx_gallery_thumbnail_number = 4

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.datasets.brainstorm import bst_resting


print(__doc__)

data_path = bst_resting.data_path()
subject = 'bst_resting'

subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')
bem_fname = op.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
src_fname = op.join(bem_dir, '%s-oct-6-src.fif' % subject)
raw_fname = (data_path + '/MEG/%s/subj002_spontaneous_20111102_01_AUX.ds'
             % subject)
raw_erm_fname = data_path + '/MEG/%s/subj002_noise_20111104_02.ds' % subject
trans_fname = data_path + '/MEG/%s/%s-trans.fif' % (subject, subject)

##############################################################################
# Load data, resample, set types, and unify channel names

# To save memory and computation time, we just use 200 sec of resting state
# data and 30 sec of empty room data

new_sfreq = 100.
raw = mne.io.read_raw_ctf(raw_fname)
raw.crop(0, 200).load_data().resample(new_sfreq)
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
raw_erm = mne.io.read_raw_ctf(raw_erm_fname)
raw_erm.crop(0, 30).load_data().resample(new_sfreq)
raw_erm.rename_channels(lambda x: x.replace('-4408', '-4407'))

##############################################################################
# Do some minimal artifact rejection

ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, tmin=-0.1, tmax=0.1,
                                                n_mag=2)
raw.add_proj(ssp_ecg)
ssp_ecg_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_mag=2)
raw.add_proj(ssp_ecg_eog, remove_existing=True)
raw_erm.add_proj(ssp_ecg_eog)

##############################################################################
# Explore data

# Alpha peak @ 8 Hz (and likely harmonic @ 16 Hz)
n_fft = int(round(4 * new_sfreq))
fig = raw.plot_psd(n_fft=n_fft, proj=True)

##############################################################################
# Make forward stack and get transformation matrix

src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)

# check alignment
mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir,
    dig=True)
fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem, eeg=False)

##############################################################################
# Compute and apply inverse

noise_cov = mne.compute_raw_covariance(raw_erm, method='shrunk')

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov)

stc_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9., method='MNE', n_fft=n_fft)
stc_psd_amp = stc_psd.copy()
stc_psd_amp.data = 10 ** (stc_psd_amp.data / 10.)

# Group into frequency bands, then normalize each source point independently
freqs = dict(delta=(2, 4), theta=(5, 7), alpha=(8, 12),
             beta=(15, 29), gamma=(30, 50))
stcs = dict()
norm = 0.
for band, limits in freqs.items():
    stcs[band] = stc_psd_amp.copy().crop(*limits).mean()
    norm += stcs[band].data
# Normalize each source point by the total power across freqs
for band in freqs.keys():
    stcs[band] /= norm

stc_psd_norm = stc_psd.copy()
stc_psd_norm.data = 10 ** (stc_psd.data / 10.)
stc_psd_norm = stc_psd_norm / stc_psd_norm.mean()

###############################################################################
# Theta:

brain_theta = stcs['theta'].plot(
    subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
    time_label="Theta (%d-%d Hz)" % freqs['theta'], title=u'Relative θ power',
    clim=dict(kind='percent', lims=(70, 85, 99)))
brain_theta.show_view(dict(azimuth=0, elevation=0), roll=0)

###############################################################################
# Alpha:

brain_alpha = stcs['alpha'].plot(
    subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
    time_label="Alpha (%d-%d Hz)" % freqs['alpha'], title=u'Relative α power',
    clim=dict(kind='percent', lims=(70, 85, 99)))
brain_alpha.show_view(dict(azimuth=0, elevation=0), roll=0)

###############################################################################
# Beta:

brain_beta = stcs['beta'].plot(
    subject=subject, subjects_dir=subjects_dir, views='dor', hemi='both',
    time_label=u"Beta (%d-%d Hz)" % freqs['beta'], title=u'Relative β power',
    clim=dict(kind='percent', lims=(70, 85, 99)))
brain_beta.show_view(dict(azimuth=0, elevation=0), roll=0)

###############################################################################
# Gamma:

brain_delta = stcs['gamma'].plot(
    subject=subject, subjects_dir=subjects_dir, views='dor', hemi='both',
    time_label="Gamma (%d-%d Hz)" % freqs['gamma'], title=u'Relative γ power',
    clim=dict(kind='percent', lims=(70, 85, 99)))
brain_delta.show_view(dict(azimuth=0, elevation=0), roll=0)
