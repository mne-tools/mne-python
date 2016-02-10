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
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

# TODO:
# - figure out why there is contamination at ~350 ms in standard condition
# - complete source localization

import numpy as np
from scipy.io import loadmat

import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import Raw

print(__doc__)

data_path = bst_auditory.data_path()

tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)
subject = 'bst_auditory'
subjects_dir = data_path + '/subjects'
raw_fnames = [data_path + '/MEG/bst_auditory/S01_AEF_20131218_0%s_raw.fif'
              % run_num for run_num in range(1, 3)]
erm_fname = data_path + '/MEG/bst_auditory/S01_Noise_20131218_01_raw.fif'

raw = Raw(raw_fnames, preload=True, add_eeg_ref=False)
raw_erm = Raw(erm_fname, preload=True, add_eeg_ref=False)

# set EOG channel
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

# leave eeg out
raw.pick_types(meg=True, eeg=False, stim=True, misc=True, eog=True, ecg=True)

# parse Brainstorm bad event data
bad_segments = list()
saccades = list()
for ii in range(1, 3):
    bad_events = loadmat(data_path + '/MEG/bst_auditory/events_bad_0%s' % ii)
    bad_events = bad_events['events']
    these_bads = these_saccades = np.empty((0, 3), int)
    for events in bad_events[0]:
        if events[0][0] == 'BAD':
            events = events[3].astype(int).T
            for event in events:
                event[1] -= event[0]
            events = np.column_stack((events, np.ones(len(events))))
            these_bads = events
        else:  # events[0][0] == 'saccade':
            events = events[3][0].astype(int)
            events = np.column_stack((events,
                                      np.zeros_like(events),
                                      np.ones_like(events)))
            these_saccades = events
    bad_segments.append(these_bads)
    saccades.append(these_saccades)

sfreq = raw.info['sfreq']

saccades = mne.concatenate_events(saccades, raw._first_samps, raw._last_samps)
bad_segments = mne.concatenate_events(bad_segments, raw._first_samps,
                                      raw._last_samps)

onsets = np.concatenate([bad_segments[:, 0] / sfreq,
                         saccades[:, 0] / sfreq])
durations = np.concatenate([bad_segments[:, 1] / sfreq,
                            saccades[:, 1] / sfreq])
bads = np.repeat('BAD', len(bad_segments))
sacs = np.repeat('saccade', len(saccades))
descriptions = np.concatenate([bads, sacs])
annot = mne.annotations.Annotations(onsets, durations, descriptions)
raw.annotations = annot
# compute saccade and EOG projectors and add them to the data
raw_filt = raw.copy()
raw_filt.filter(1., 15.)
saccade_epochs = mne.Epochs(raw_filt, saccades, 1, 0., 0.5, preload=True,
                            segment_reject=False)

projs_saccade = mne.compute_proj_epochs(saccade_epochs, n_mag=1, n_eeg=0,
                                        desc_prefix='saccade')

del raw_filt, saccade_epochs  # to save memory
projs_eog = mne.preprocessing.compute_proj_eog(raw, n_mag=1, n_eeg=0)[0]
raw.add_proj(projs_saccade)
raw.add_proj(projs_eog)

# look at the effects of projection
raw.plot()

# show power line interference and remove it
meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
raw.plot_psd(picks=meg_picks)
notches = np.arange(60, 181, 60)
raw.notch_filter(notches)
raw_erm.notch_filter(notches)
raw.plot_psd(picks=meg_picks)

# lowpass data at 100 Hz
raw.filter(None, 100.)
raw_erm.filter(None, 100.)

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
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False,
                    proj=True)

# compute evoked
evoked_standard = epochs['standard'].average()
evoked_deviant = epochs['deviant'].average()

# plot the result
evoked_standard.plot(window_title='Standard', gfp=True)
evoked_deviant.plot(window_title='Deviant', gfp=True)

# show topomaps
times = np.arange(0, 0.351, 0.025)
evoked_standard.plot_topomap(times=times)
evoked_deviant.plot_topomap(times=times)

evoked_difference = evoked_standard.copy()
evoked_difference.data = evoked_deviant.data - evoked_standard.data
evoked_difference.plot(window_title='Difference', gfp=True)

# compute noise covariance
cov = mne.compute_raw_covariance(raw_erm, reject=reject)
cov.plot(raw_erm.info)

src = mne.read_source_spaces(subjects_dir +
                             '/bst_auditory/bem/bst_auditory-oct-6-src.fif')
trans = mne.read_trans(data_path + '/MEG/bst_auditory/mainak-trans.fif')
surfs = mne.read_bem_surfaces(subjects_dir +
                              '/bst_auditory/bem/bst_auditory-inner-skull.fif')
bem = mne.make_bem_solution(surfs)

fwd = mne.make_forward_solution(evoked_standard.info, trans=trans, src=src,
                                bem=bem)

inv = mne.minimum_norm.make_inverse_operator(evoked_standard.info, fwd, cov)

snr = 3.0
lambda2 = 1.0 / snr ** 2
stc_standard = mne.minimum_norm.apply_inverse(evoked_standard, inv, lambda2,
                                              'dSPM')
stc_deviant = mne.minimum_norm.apply_inverse(evoked_deviant, inv, lambda2,
                                             'dSPM')
stc_difference = mne.minimum_norm.apply_inverse(evoked_difference, inv,
                                                lambda2, 'dSPM')

stc_standard.plot(subjects_dir=subjects_dir, subject=subject,
                  surface='inflated', time_viewer=True, hemi='both')

stc_deviant.plot(subjects_dir=subjects_dir, subject=subject,
                 surface='inflated', time_viewer=True, hemi='both')

stc_difference.plot(subjects_dir=subjects_dir, subject=subject,
                    surface='inflated', time_viewer=True, hemi='both')
