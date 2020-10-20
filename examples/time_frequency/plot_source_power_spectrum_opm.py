"""
.. _ex-opm-resting-state:

======================================================================
Compute source power spectral density (PSD) of VectorView and OPM data
======================================================================

Here we compute the resting state from raw for data recorded using
a Neuromag VectorView system and a custom OPM system.
The pipeline is meant to mostly follow the Brainstorm [1]_
`OMEGA resting tutorial pipeline <bst_omega_>`_.
The steps we use are:

1. Filtering: downsample heavily.
2. Artifact detection: use SSP for EOG and ECG.
3. Source localization: dSPM, depth weighting, cortically constrained.
4. Frequency: power spectral density (Welch), 4 sec window, 50% overlap.
5. Standardize: normalize by relative power for each source.

.. contents::
   :local:
   :depth: 1

.. _bst_omega: https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega

Preprocessing
-------------
"""
# sphinx_gallery_thumbnail_number = 13

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mne.filter import next_fast_len

import mne


print(__doc__)

data_path = mne.datasets.opm.data_path()
subject = 'OPM_sample'

subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')
bem_fname = op.join(subjects_dir, subject, 'bem',
                    subject + '-5120-5120-5120-bem-sol.fif')
src_fname = op.join(bem_dir, '%s-oct6-src.fif' % subject)
vv_fname = data_path + '/MEG/SQUID/SQUID_resting_state.fif'
vv_erm_fname = data_path + '/MEG/SQUID/SQUID_empty_room.fif'
vv_trans_fname = data_path + '/MEG/SQUID/SQUID-trans.fif'
opm_fname = data_path + '/MEG/OPM/OPM_resting_state_raw.fif'
opm_erm_fname = data_path + '/MEG/OPM/OPM_empty_room_raw.fif'
opm_trans_fname = None
opm_coil_def_fname = op.join(data_path, 'MEG', 'OPM', 'coil_def.dat')

##############################################################################
# Load data, resample. We will store the raw objects in dicts with entries
# "vv" and "opm" to simplify housekeeping and simplify looping later.

raws = dict()
raw_erms = dict()
new_sfreq = 90.  # Nyquist frequency (45 Hz) < line noise freq (50 Hz)
raws['vv'] = mne.io.read_raw_fif(vv_fname, verbose='error')  # ignore naming
raws['vv'].load_data().resample(new_sfreq)
raws['vv'].info['bads'] = ['MEG2233', 'MEG1842']
raw_erms['vv'] = mne.io.read_raw_fif(vv_erm_fname, verbose='error')
raw_erms['vv'].load_data().resample(new_sfreq)
raw_erms['vv'].info['bads'] = ['MEG2233', 'MEG1842']

raws['opm'] = mne.io.read_raw_fif(opm_fname)
raws['opm'].load_data().resample(new_sfreq)
raw_erms['opm'] = mne.io.read_raw_fif(opm_erm_fname)
raw_erms['opm'].load_data().resample(new_sfreq)
# Make sure our assumptions later hold
assert raws['opm'].info['sfreq'] == raws['vv'].info['sfreq']

##############################################################################
# Do some minimal artifact rejection just for VectorView data

titles = dict(vv='VectorView', opm='OPM')
ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(
    raws['vv'], tmin=-0.1, tmax=0.1, n_grad=1, n_mag=1)
raws['vv'].add_proj(ssp_ecg, remove_existing=True)
# due to how compute_proj_eog works, it keeps the old projectors, so
# the output contains both projector types (and also the original empty-room
# projectors)
ssp_ecg_eog, _ = mne.preprocessing.compute_proj_eog(
    raws['vv'], n_grad=1, n_mag=1, ch_name='MEG0112')
raws['vv'].add_proj(ssp_ecg_eog, remove_existing=True)
raw_erms['vv'].add_proj(ssp_ecg_eog)
fig = mne.viz.plot_projs_topomap(raws['vv'].info['projs'][-4:],
                                 info=raws['vv'].info)
fig.suptitle(titles['vv'])
fig.subplots_adjust(0.05, 0.05, 0.95, 0.85)

##############################################################################
# Explore data

kinds = ('vv', 'opm')
n_fft = next_fast_len(int(round(4 * new_sfreq)))
print('Using n_fft=%d (%0.1f sec)' % (n_fft, n_fft / raws['vv'].info['sfreq']))
for kind in kinds:
    fig = raws[kind].plot_psd(n_fft=n_fft, proj=True)
    fig.suptitle(titles[kind])
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)

##############################################################################
# Alignment and forward
# ---------------------

# Here we use a reduced size source space (oct5) just for speed
src = mne.setup_source_space(
    subject, 'oct5', add_dist=False, subjects_dir=subjects_dir)
# This line removes source-to-source distances that we will not need.
# We only do it here to save a bit of memory, in general this is not required.
del src[0]['dist'], src[1]['dist']
bem = mne.read_bem_solution(bem_fname)
fwd = dict()

# check alignment and generate forward for VectorView
kwargs = dict(azimuth=0, elevation=90, distance=0.6, focalpoint=(0., 0., 0.))
fig = mne.viz.plot_alignment(
    raws['vv'].info, trans=vv_trans_fname, subject=subject,
    subjects_dir=subjects_dir, dig=True, coord_frame='mri',
    surfaces=('head', 'white'))
mne.viz.set_3d_view(figure=fig, **kwargs)
fwd['vv'] = mne.make_forward_solution(
    raws['vv'].info, vv_trans_fname, src, bem, eeg=False, verbose=True)

##############################################################################
# And for OPM:

with mne.use_coil_def(opm_coil_def_fname):
    fig = mne.viz.plot_alignment(
        raws['opm'].info, trans=opm_trans_fname, subject=subject,
        subjects_dir=subjects_dir, dig=False, coord_frame='mri',
        surfaces=('head', 'white'))
    mne.viz.set_3d_view(figure=fig, **kwargs)
    fwd['opm'] = mne.make_forward_solution(
        raws['opm'].info, opm_trans_fname, src, bem, eeg=False, verbose=True)

del src, bem

##############################################################################
# Compute and apply inverse to PSD estimated using multitaper + Welch.
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(
    delta=(2, 4), theta=(5, 7), alpha=(8, 12), beta=(15, 29), gamma=(30, 45))
topos = dict(vv=dict(), opm=dict())
stcs = dict(vv=dict(), opm=dict())

snr = 3.
lambda2 = 1. / snr ** 2
for kind in kinds:
    noise_cov = mne.compute_raw_covariance(raw_erms[kind])
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raws[kind].info, forward=fwd[kind], noise_cov=noise_cov, verbose=True)
    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raws[kind], inverse_operator, lambda2=lambda2,
        n_fft=n_fft, dB=False, return_sensor=True, verbose=True)
    topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
    stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
    # Normalize each source point by the total power across freqs
    for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[kind][band] = mne.EvokedArray(
            100 * data / topo_norm, sensor_psd.info)
        stcs[kind][band] = \
            100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data
    del inverse_operator
del fwd, raws, raw_erms


###############################################################################
# Now we can make some plots of each frequency band. Note that the OPM head
# coverage is only over right motor cortex, so only localization
# of beta is likely to be worthwhile.
#
# Theta
# -----

def plot_band(kind, band):
    """Plot activity within a frequency band on the subject's brain."""
    title = "%s %s\n(%d-%d Hz)" % ((titles[kind], band,) + freq_bands[band])
    topos[kind][band].plot_topomap(
        times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
        time_format=title)
    brain = stcs[kind][band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=title, colormap='inferno',
        time_viewer=False, show_traces=False,
        clim=dict(kind='percent', lims=(70, 85, 99)), smoothing_steps=10)
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain


fig_theta, brain_theta = plot_band('vv', 'theta')

###############################################################################
# Alpha
# -----

fig_alpha, brain_alpha = plot_band('vv', 'alpha')

###############################################################################
# Beta
# ----
# Here we also show OPM data, which shows a profile similar to the VectorView
# data beneath the sensors. VectorView first:

fig_beta, brain_beta = plot_band('vv', 'beta')

###############################################################################
# Then OPM:
fig_beta_opm, brain_beta_opm = plot_band('opm', 'beta')

###############################################################################
# Gamma
# -----

fig_gamma, brain_gamma = plot_band('vv', 'gamma')

###############################################################################
# References
# ----------
# .. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
#        Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
#        Computational Intelligence and Neuroscience, vol. 2011, Article ID
#        879716, 13 pages, 2011. doi:10.1155/2011/879716
