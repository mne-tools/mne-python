"""
================================
Brainstorm resting state dataset
================================

Here we compute the resting state from raw for the
Brainstorm tutorial dataset. For comparison, see [1]_ and:

    http://neuroimage.usc.edu/brainstorm/Tutorials/MedianNerveCtf

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from scipy import stats

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
# Load data, set types and rename ExG channels

raw = mne.io.read_raw_ctf(raw_fname, preload=True)
raw_erm = mne.io.read_raw_ctf(raw_erm_fname, preload=True)

# clean up bad ch names and do common preprocessing
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
for raw_ in (raw, raw_erm):
    raw_.resample(100, n_jobs=2)

# unify channel names
raw_erm.rename_channels(lambda x: x.replace('-4408', '-4407'))

##############################################################################
# Compute SSP

ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, n_mag=2)
ssp_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_mag=2)

raw.add_proj(ssp_eog + ssp_ecg)
raw_erm.add_proj(ssp_eog + ssp_ecg)

##############################################################################
# Explore data

raw.plot_psd(n_fft=2048, fmin=1, fmax=50, xscale='log', proj=True)

# we see some weakly pronounced peak around 8-9 Hz and some wider beta band
# activity peaking at 16 Hz. What could the underlying brain
# sources look like?

##############################################################################
# Make forward stack and get transformation matrix

src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)
picks = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=True)
trans = mne.read_trans(trans_fname)
mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir)
fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem,
                                eeg=False, verbose=True)

##############################################################################
# Make epochs and look at power

overlap = 4
tmax = 12
reject = dict(mag=5e-12)

events = mne.make_fixed_length_events(raw, id=42, duration=overlap)

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       ref_meg=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id=42, tmin=0, tmax=tmax, picks=picks,
                    baseline=None, reject=reject, preload=False,
                    proj=True)

# epochs.plot_psd_topomap(normalize=True)  # very slow!

##############################################################################
# Make inverse

noise_cov = mne.compute_raw_covariance(raw_erm, tmax=30)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    epochs.info, forward=fwd, noise_cov=noise_cov)

stc_gen = mne.minimum_norm.apply_inverse_epochs(
    epochs, inverse_operator, lambda2=1., method='MNE', nave=1,
    pick_ori="normal", return_generator=True, prepared=False)

# make PSD in source sapce
psd_src = list()
for ii, this_stc in enumerate(stc_gen):
    psd, freqs = mne.time_frequency.psd_array_welch(
        this_stc.data, sfreq=epochs.info['sfreq'],
        n_fft=1024, fmin=1, fmax=50)
# changing this makes a big difference.
#    psd = np.log10(psd)

    if True:  # compute relative power
        psd /= psd.sum(axis=1, keepdims=True)
    psd_src.append(psd)
psd_src = np.mean(psd_src, axis=0)

if False:
    # normalize each frequency bin across space to deal with color map.
    # and make sure it's positive.
    psd_src = 10 + stats.zscore(psd_src, axis=0)

# make STC where time is frequency.
stc_psd = mne.SourceEstimate(
    data=psd_src,
    subject=subject,
    vertices=[fwd['src'][ii]['vertno'] for ii in [0, 1]],
    tmin=freqs[0],
    tstep=np.diff(freqs)[0])

# crop to the frequency of interest to satisfy colormap mechanism
stc_psd.copy().crop(8, 8 + stc_psd.tstep).plot(
    subject=subject,
    subjects_dir=subjects_dir,
    views='cau',
    hemi='both',
    time_viewer=True,
    time_label="Freq=%0.2f Hz",
    colormap='viridis',
    clim=dict(kind='percent', lims=(0, 80, 99))
)
