# -*- coding: utf-8 -*-
"""
====================================
Brainstorm auditory tutorial dataset
====================================

Here we compute the evoked from raw for the auditory Brainstorm
tutorial dataset. For comparison, see:
http://neuroimage.usc.edu/brainstorm/Tutorials/Auditory

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

# TODO:
# - remove saccade artifacts (likely source of ~350ms contamination)
# - complete source localization

import numpy as np

import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import Raw

print(__doc__)

tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)

data_path = bst_auditory.data_path()

raw_fnames = [data_path + '/MEG/bst_auditory/S01_AEF_20131218_0%s_raw.fif'
              % run_num for run_num in range(1, 3)]
raw = Raw(raw_fnames, preload=True)

# set EOG channel
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

# compute EOG and ECG projectors and add them to the data
projs_eog = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1,
                                               n_eeg=0)[0]
raw.add_proj(projs_eog)
projs_ecg = mne.preprocessing.compute_proj_eog(raw, n_grad=1, n_mag=1,
                                               n_eeg=0)[0]
raw.add_proj(projs_ecg)

# show power line interference and remove it
meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
raw.plot_psd(picks=meg_picks)
raw.notch_filter(np.arange(60, 181, 60))
raw.plot_psd(picks=meg_picks)

# lowpass data at 100 Hz
raw.filter(None, 100.)

# find events
events = mne.find_events(raw, stim_channel='UPPT001')

# adjust event times based on detected sound onsets
sound_data = raw[mne.pick_channels(raw.ch_names, ['UADC001-4408'])][0][0]
onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
min_diff = int(0.5 * raw.info['sfreq'])
diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
onsets = onsets[diffs > min_diff]
assert len(onsets) == len(events)
diffs = 1000. * (events[:, 0] - onsets) / raw.info['sfreq']
print('Trigger delay removed (μ ± σ): %0.1f ± %0.1f ms'
      % (np.mean(diffs), np.std(diffs)))
events[:, 0] = onsets

# set bad channels
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

# compute evoked
evoked_standard = epochs['standard'].average()
evoked_deviant = epochs['deviant'].average()

# plot the result
evoked_standard.plot(window_title='Standard', gfp=True)
evoked_deviant.plot(window_title='Deviant', gfp=True)

# show topomaps
evoked_standard.plot_topomap(times=np.array([0.095]))
evoked_deviant.plot_topomap(times=np.array([0.095]))

# compute noise covariance
# cov = mne.compute_covariance(epochs, method='shrunk')

# compute forward
# fwd = mne.make_forward_solution(evoked_standard.info, trans, src, bem)

# compute inverse
# inv = mne.minimum_norm.make_inverse_operator(evoked_standard.info, fwd, cov)

# apply inverse
# stc = mne.minimum_norm.apply_inverse(evoked_standard, inv)

# plot result
# stc.plot(views=['lat', 'med'], hemi='split')
