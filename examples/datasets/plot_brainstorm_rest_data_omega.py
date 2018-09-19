"""
================================
Brainstorm resting state dataset
================================

Here we compute the resting state from raw for the
Brainstorm tutorial dataset see [1]_

The pipeline is meant to mirror the brainstorm resting tutorial pipeline

    http://neuroimage.usc.edu/brainstorm/Tutorials/MedianNerveCtf

Brainstorm pipline is
1) Notch filter: 60 120 180 240 300 Hz,
2) Band-pass filter: High-pass filter at 0.3Hz
3) Artifact (EOG/ECG) detection... XXX skipped
4) Source localization (dSPM, some depth weighting, constrained)
5) Frequency > Power spectrum density (Welch): [0,100s], Window=4s, 50% overlap,
        Group in frequency bands (use the default frequency bands)
6) Standardize > Spectrum normalization: Relative power (divide by total power)

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#
# License: BSD (3-clause)

import mayavi.mlab as mlab
import os.path as op

import numpy as np

import mne
from mne.datasets.brainstorm import bst_resting


print(__doc__)

data_path = bst_resting.data_path()
subject = 'bst_resting'

subjects_dir = op.join(data_path, 'subjects')

raw_fname = data_path + '/MEG/' + subject + '/' + \
                        'subj002_spontaneous_20111102_01_AUX_raw.fif'

raw_noise_fname = op.join(
    data_path,
    'MEG/%s/subj002_noise_20111104_02_raw.fif' % subject)

# XXX trans should be replaced by file in dataset.
# this was not guessed but is from a file that we don't ship.
trans = mne.Transform(  # Kids please do not do this at home.
    fro=4,  # head
    to=5,  # mri
    trans=np.array(  # run 1
        [[0.99979711, 0.01138957, 0.01660261, -0.00337221],
         [-0.00577519, 0.95219451, -0.30543712, -0.00715041],
         [-0.01928758, 0.30527931, 0.95206732, -0.03214351],
         [0., 0., 0, 1.]]))

##############################################################################
# Load data, set types and rename ExG channels

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_er = mne.io.read_raw_fif(raw_noise_fname, preload=True)

# raw.crop(0, 60)

# clean up bad ch names and do common preprocessing
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
for raw_ in (raw, raw_er):
    raw.apply_gradient_compensation(3)
    raw_.resample(300, n_jobs=2)
    raw_.rename_channels(
        dict(zip(raw_.ch_names, mne.utils._clean_names(raw_.ch_names))))
    picks = mne.pick_types(raw_.info, meg=True, eog=True, ecg=True)
    raw_.notch_filter([60., 120.], filter_length='auto', phase='zero')
    raw_.filter(0.3, None)

for comp in raw.info['comps']:
    for key in ('row_names', 'col_names'):
        comp['data'][key] = mne.utils._clean_names(comp['data'][key])

##############################################################################
# Explore data

raw.plot_psd(n_fft=2048, fmin=1, fmax=100, proj=True)

# we see some weakly pronounced peak around 8-9 Hz and some wider beta band
# activity peaking at 16 Hz. What could the underlying brain
# sources look like?

##############################################################################
# Make forward stack and get transformation matrix

spacing = 'oct6'
src = mne.setup_source_space(
    subject=subject, spacing=spacing, subjects_dir=subjects_dir,
    add_dist=False)


conductivity = (0.3,)  # for single layer
model = mne.make_bem_model(subject='bst_resting', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem,
                                eeg=False)

###############################################################################
# check alignment

mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir)

##############################################################################
# Make inverse

noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', tmax=30)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov)

# compute source psd from raw
fmin, fmax = 0., 100.
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

psd_raw_db = mne.minimum_norm.compute_source_psd(raw, inverse_operator,
                                                 lambda2=lambda2,
                                                 method=method,
                                                 fmin=fmin, fmax=fmax,
                                                 n_fft=2048, label=None,
                                                 out_decibels=True)


psd_raw = mne.minimum_norm.compute_source_psd(raw, inverse_operator,
                                              lambda2=lambda2,
                                              method=method,
                                              fmin=fmin, fmax=fmax,
                                              n_fft=2048, label=None,
                                              out_decibels=False)

# compute total and relative power.
psd_total_raw_db = psd_raw_db.mean() * len(psd_raw_db.times)
psd_rel_raw_db = psd_raw_db / psd_total_raw_db

psd_total_raw = psd_raw.mean() * len(psd_raw.times)
psd_rel_raw = psd_raw / psd_total_raw

###############################################################################
# collect into frequency bands.
freq_bands = [(2, 4, 'delta'), (5, 7, 'theta'), (8, 12, 'alpha'),
              (15, 29, 'beta'), (30, 59, 'gamma_1'), (60, 90, 'gamma_2')]

stc_bands_db = dict()
stc_bands = dict()
for r_min, r_max, b in freq_bands:
    f_min = 1e-3 * r_min
    f_max = 1e-3 * r_max
    stc_tmp = psd_rel_raw_db.copy().crop(f_min, f_max)
    stc_bands_db[b] = stc_tmp.mean() * len(stc_tmp.times)

    stc_tmp = psd_rel_raw.copy().crop(f_min, f_max)
    stc_bands[b] = stc_tmp.mean() * len(stc_tmp.times)

##############################################################################
# plot beta. notice the ventral surface is the peak activity with DB
b = 'beta'
for stc, u in zip([stc_bands[b], stc_bands_db[b]], ['Not DB', 'DB']):
    fig = mlab.figure(size=(800, 600))
    mlab.clf()
    brain = stc.plot(subject, 'pial', 'both', figure=fig,
                     colorbar=True, colormap='jet',
                     subjects_dir=subjects_dir,
                     clim=dict(kind='percent', lims=(0, 50, 100)),
                     views=['ven'])
    mlab.title('rel beta power - %s' % u)

mlab.show()

##############################################################################
# plot relative power in each band
# for _, _, b in freq_bands:
#     fig = mlab.figure(size=(800, 600))
#     mlab.clf()
#     brain = stc_bands[b].plot(subject, 'pial', 'both', figure=fig,
#                               colorbar=True, colormap='jet',
#                               subjects_dir=subjects_dir,
#                               clim=dict(kind='percent', lims=(0, 50, 100)))
#     mlab.title('%s - power' % b)
# mlab.show()
#
#
# ##############################################################################
# # crop to the frequency of interest to satisfy colormap mechanism
#
# f = 1e-3 * 8
# psd_rel_raw.copy().crop(f, f + psd_rel_raw.tstep).mean().plot(
#     subject=subject,
#     subjects_dir=subjects_dir,
#     views='cau',
#     hemi='both',
#     time_label="Freq=%0.4f Hz",
#     colormap='viridis',
#     clim=dict(kind='percent', lims=(0, 80, 99))
# )
# mlab.show()
#
