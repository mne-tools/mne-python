"""
"""
# sphinx_gallery_thumbnail_number = 4

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mne.filter import next_fast_len
import matplotlib.pyplot as plt
from mayavi import mlab

import mne


print(__doc__)

data_path = mne.datasets.opm.data_path()
subject = 'OPM_sample'

subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, subject, 'bem')
bem_fname = op.join(subjects_dir, subject, 'bem',
                    subject + '-5120-5120-5120-bem-sol.fif')
src_fname = op.join(bem_dir, '%s-oct6-src.fif' % subject)
raw_fname = data_path + '/MEG/SQUID/SQUID_resting_state.fif'
raw_erm_fname = data_path + '/MEG/SQUID/SQUID_empty_room.fif'
trans_fname = data_path + '/MEG/SQUID/SQUID-trans.fif'

################
_USE_DB = False
################

##############################################################################
# Load data, resample, set types, and unify channel names

# To save memory and computation time, we just use 60 sec of resting state
# data and 30 sec of empty room data.

new_sfreq = 100.
raw = mne.io.read_raw_fif(raw_fname)
raw.crop(0, None).load_data().resample(new_sfreq)
raw_erm = mne.io.read_raw_fif(raw_erm_fname)
raw_erm.crop(0, None).load_data().resample(new_sfreq)
raw.info['bads'] += ['MEG2233']
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

noise_cov = mne.compute_raw_covariance(raw_erm, method='oas')

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov, verbose=True)

stc_psd, evoked_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9., method='MNE', n_fft=n_fft,
    dB=_USE_DB, return_sensor=True, verbose=True)
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

mag_picks = mne.pick_types(evoked_psd.info, meg='mag')
grad_picks = mne.pick_types(evoked_psd.info, meg='grad')

mag_info = mne.pick_info(evoked_psd.info, sel=mag_picks, copy=True)
grad_info = mne.pick_info(evoked_psd.info, sel=grad_picks, copy=True)

def plot_band_old(band):
    title = "%s (%d-%d Hz) - db=%s" % ((band.capitalize(),) + freqs[band] +
                                       (_USE_DB,))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    vmin_mag, vmax_mag, vmin_grad, vmax_grad = (None, None, None, None)
    if _USE_DB:
        vmin_mag = topos[band][mag_picks].min()
        vmax_mag = topos[band][mag_picks].max()
        vmin_grad = topos[band][grad_picks].min()
        vmax_grad = topos[band][grad_picks].max()
    mne.viz.plot_topomap(topos[band][mag_picks], mag_info,
                         vmin=vmin_mag, vmax=vmax_mag,
                         outlines='skirt', axes=ax[0], show=False)
    mne.viz.plot_topomap(topos[band][grad_picks], grad_info,
                         vmin=vmin_grad, vmax=vmax_grad,
                         outlines='skirt', axes=ax[1], show=False)
    ax[0].set_xlabel(title)
    ax[1].set_xlabel(title)
    fig.tight_layout()
    brain = stcs[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=u'Relative %s power' % band,
        clim=dict(kind='percent', lims=(70, 85, 99)))
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain


def plot_band(band):
    title = "%s (%d-%d Hz) - db=%s" % ((band.capitalize(),) + freq_bands[band]+
                                       (_USE_DB,))
    topos[band].plot_topomap(
        times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
        time_format=title)
    brain = stcs[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=title, colormap='inferno',
        clim=dict(kind='percent', lims=(70, 85, 99)))
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain

###############################################################################
# Theta:

fig_theta, brain_theta = plot_band('theta')
plt.show()
###############################################################################
# Alpha:

fig_alpha, brain_alpha = plot_band('alpha')

###############################################################################
# Beta:

fig_beta, brain_beta = plot_band('beta')

###############################################################################
# Gamma:

fig_gamma, brain_gamma = plot_band('gamma')
plt.show()
