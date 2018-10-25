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

raw_fname = (data_path + '/MEG/%s/subj002_spontaneous_20111102_01_AUX.ds'
             % subject)

raw_noise_fname = data_path + '/MEG/%s/subj002_noise_20111104_02.ds' % subject

trans_fname = data_path + '/MEG/%s/%s-trans.fif' % (subject, subject)


##############################################################################
# Load data, set types and rename ExG channels

raw = mne.io.read_raw_ctf(raw_fname, preload=True)
raw_er = mne.io.read_raw_ctf(raw_noise_fname, preload=True)

# clean up bad ch names and do common preprocessing
raw.set_channel_types({'EEG057': 'ecg', 'EEG058': 'eog'})
for raw_ in (raw, raw_er):
    raw_.resample(150, n_jobs=2)
    raw_.rename_channels(
        dict(zip(raw_.ch_names, mne.utils._clean_names(raw_.ch_names))))
    picks = mne.pick_types(raw_.info, meg=True, eog=True, ecg=True)
    raw_.filter(1, None)

for comp in raw.info['comps']:
    for key in ('row_names', 'col_names'):
        comp['data'][key] = mne.utils._clean_names(comp['data'][key])

##############################################################################
# Compute SSP

ssp_ecg, _ = mne.preprocessing.compute_proj_ecg(
    raw, average=True, n_mag=2)
ssp_eog, _ = mne.preprocessing.compute_proj_eog(
    raw, average=True, n_mag=2)

raw.add_proj(ssp_eog)
raw.add_proj(ssp_ecg)

raw_er.add_proj(ssp_eog)
raw_er.add_proj(ssp_ecg)


##############################################################################
# Explore data

raw.plot_psd(n_fft=2048, fmin=1, fmax=50, xscale='log', proj=True)

# we see some weakly pronounced peak around 8-9 Hz and some wider beta band
# activity peaking at 16 Hz. What could the underlying brain
# sources look like?

##############################################################################
# Make forward stack and get transformation matrix

spacing = 'oct6'
src = mne.setup_source_space(
    subject=subject, spacing=spacing, subjects_dir=subjects_dir,
    add_dist=True)

conductivity = (0.3,)  # for single layer
model = mne.make_bem_model(subject='bst_resting', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
raise RuntimeError

picks = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=True)


trans = mne.read_trans(trans_fname)
mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir)

fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem,
                                eeg=False)

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

epochs.plot_psd_topomap(normalize=True)


##############################################################################
# Make inverse

noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', tmax=30)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    epochs.info, forward=fwd, noise_cov=noise_cov)


stc_gen = mne.minimum_norm.apply_inverse_epochs(
    epochs, inverse_operator,
    lambda2=1,
    method='MNE', nave=1, pick_ori="normal",
    return_generator=True, prepared=False)

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
