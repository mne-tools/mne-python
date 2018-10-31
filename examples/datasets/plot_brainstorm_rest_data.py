"""
================================
Brainstorm resting state dataset
================================

Here we compute the resting state from raw for the
Brainstorm tutorial dataset, see [1]_.

The pipeline is meant to mirror the Brainstorm
`OMEGA resting tutorial pipeline <bst_omega_>`_, though these data come
from the Brainstorm `resting dataset <bst_resting_>`_. The steps we use are:

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

.. _bst_omega: https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega
.. _bst_resting: https://neuroimage.usc.edu/brainstorm/Tutorials/Resting
"""
# sphinx_gallery_thumbnail_number = 5

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mne.filter import next_fast_len
from mayavi import mlab

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

# To save memory and computation time, we just use 60 sec of resting state
# data and 30 sec of empty room data.

new_sfreq = 100.
raw = mne.io.read_raw_ctf(raw_fname)
raw.crop(220, 280).load_data().resample(new_sfreq)
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
raw_erm = mne.io.read_raw_ctf(raw_erm_fname)
raw_erm.crop(0, 30).load_data().resample(new_sfreq)
raw_erm.rename_channels(lambda x: x.replace('-4408', '-4407'))

##############################################################################
# Do some minimal artifact rejection

ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, tmin=-0.1, tmax=0.1,
                                                n_mag=2)
raw.add_proj(ssp_ecg)
ssp_ecg_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_mag=1)
raw.add_proj(ssp_ecg_eog, remove_existing=True)
raw_erm.add_proj(ssp_ecg_eog)
mne.viz.plot_projs_topomap(raw.info['projs'], info=raw.info)

##############################################################################
# Explore data

n_fft = next_fast_len(int(round(4 * new_sfreq)))
print('Using n_fft=%d (%0.1f sec)' % (n_fft, n_fft / raw.info['sfreq']))
raw.plot_psd(n_fft=n_fft, proj=True)

##############################################################################
# Make forward stack and get transformation matrix

src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)

# check alignment
fig = mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir,
    dig=True, coord_frame='meg')
mlab.view(180, 90, figure=fig)

fwd = mne.make_forward_solution(
    raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)

##############################################################################
# Compute and apply inverse to PSD estimated using multitaper + Welch

noise_cov = mne.compute_raw_covariance(raw_erm)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov, verbose=True)

stc_psd, evoked_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9., method='MNE', n_fft=n_fft,
    dB=False, return_sensor=True, verbose=True)

##############################################################################
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(
    delta=(2, 4), theta=(5, 7), alpha=(8, 12), beta=(15, 29), gamma=(30, 50))
topos = dict()
stcs = dict()
topo_norm = evoked_psd.data.sum(axis=1, keepdims=True)
stc_norm = stc_psd.sum()
# Normalize each source point by the total power across freqs
for band, limits in freq_bands.items():
    data = evoked_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
    topos[band] = mne.EvokedArray(100 * data / topo_norm, evoked_psd.info)
    stcs[band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data

###############################################################################
# Theta:


def plot_band(band):
    title = "%s (%d-%d Hz)" % ((band.capitalize(),) + freq_bands[band])
    topos[band].plot_topomap(
        times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
        time_format=title)
    brain = stcs[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=title, colormap='inferno',
        clim=dict(kind='percent', lims=(70, 85, 99)))
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain


fig_theta, brain_theta = plot_band('theta')

###############################################################################
# Alpha:

fig_alpha, brain_alpha = plot_band('alpha')

###############################################################################
# Beta:

fig_beta, brain_beta = plot_band('beta')

###############################################################################
# Gamma:

fig_gamma, brain_gamma = plot_band('gamma')
