# -*- coding: utf-8 -*-
"""
====================================
Brainstorm auditory tutorial dataset
====================================

Here we compute the evoked from raw for the auditory Brainstorm
tutorial dataset. For comparison, see:
http://neuroimage.usc.edu/brainstorm/Tutorials/Auditory

Experiment:
    - One subject 2 acquisition runs 6 minutes each.
    - Each run contains 200 regular beeps and 40 easy deviant beeps.
    - Random ISI: between 0.7s and 1.7s seconds, uniformly distributed.
    - Button pressed when detecting a deviant with the right index finger.

The specifications of this dataset were discussed initially on the FieldTrip
bug tracker:
http://bugzilla.fcdonders.nl/show_bug.cgi?id=2300

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

import numpy as np
import os.path as op
import csv

import mne
from mne.datasets.brainstorm import bst_auditory
from mne.io import Raw

print(__doc__)


# To reduce running time, the source space analysis is omitted. To run the
# whole tutorial change this to True.
run_source_space = False

data_path = bst_auditory.data_path()

subject = 'bst_auditory'
subjects_dir = data_path + '/subjects'
raw_fnames = [data_path + '/MEG/bst_auditory/S01_AEF_20131218_0%s_raw.fif'
              % run_num for run_num in range(1, 3)]
erm_fname = data_path + '/MEG/bst_auditory/S01_Noise_20131218_01_raw.fif'

raw = Raw(raw_fnames, preload=True, add_eeg_ref=False)
raw_erm = Raw(erm_fname, preload=True, add_eeg_ref=False)

##############################################################################
# The data was collected with a CTF 275 system at 2400 Hz and low-pass
# filtered at 600 Hz. Data channel array consisted of 274 MEG axial
# gradiometers, 26 MEG reference sensors and 2 EEG electrodes (Cz and Pz).
# In addition:
#   - 1 stim channel for marking presentation times for the stimuli
#   - 1 audio channel for the sent signal
#   - 1 response channel for recording the button presses
#   - 1 ECG bipolar
#   - 2 EOG bipolar (vertical and horizontal)
#   - 12 head tracking channels
#   - 20 unused channels
# The head tracking channels and the unused channels are marked as misc
# channels. Here we define the EOG and ECG channels.
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})

# Leave out the two EEG channels for easier setup of source space.
raw.pick_types(meg=True, eeg=False, stim=True, misc=True, eog=True, ecg=True)

##############################################################################
# For noise reduction, a set of bad segments have been identified and stored
# in a csv file. The bad segments are later used to reject epochs that overlap
# with them.
# The file also contains some saccades from the second run. The saccades are
# removed by using SSP. You can view the file with your favorite text editor.

onsets = list()
durations = list()
descriptions = list()
saccades = list()
with open(data_path + '/MEG/bst_auditory/events_bad.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        onsets.append(float(row[0]))
        durations.append(float(row[1]))
        descriptions.append(row[3])
        if row[3] == 'saccade':  # Construct events for removal of saccades.
            saccades.append([int(float(row[0])), 0, 1])
saccades = np.asarray(saccades)

##############################################################################
# Annotation start times and durations are made in seconds, so conversions
# from sample numbers to seconds is needed:
onsets = raw.index_as_time(onsets)
durations = raw.index_as_time(durations)

annot = mne.annotations.Annotations(onsets, durations, descriptions)
raw.annotations = annot

##############################################################################
# Here we compute the saccade and EOG projectors for magnetometers and add
# them to the raw data.
raw_filt = raw.copy()
raw_filt.filter(1., 15.)
saccade_epochs = mne.Epochs(raw_filt, saccades, 1, 0., 0.5, preload=True,
                            segment_reject=False)

projs_saccade = mne.compute_proj_epochs(saccade_epochs, n_mag=1, n_eeg=0,
                                        desc_prefix='saccade')

del raw_filt, saccade_epochs  # to save memory
projs_eog, eog_events = mne.preprocessing.compute_proj_eog(raw, n_mag=1,
                                                           n_eeg=0)
raw.add_proj(projs_saccade)
raw.add_proj(projs_eog)

##############################################################################
# Visually inspect the effects of projections. Click on 'proj' button at the
# bottom right corner to toggle the projectors on/off. EOG events can be
# plotted by adding the event list as a keyword argument. As the bad segments
# and saccades were added as annotations to the raw data, they are plotted as
# well. You should also check that the EOG detection algorithm did it's job
# and the events are well aligned with the blinks.
raw.plot(events=eog_events)

##############################################################################
# Typical preprocessing step is the removal of power line artifact (50 Hz or
# 60 Hz). Here we notch filter the data at 60, 120 and 180 to remove the
# original 60 Hz artifact and the harmonics. The power spectra are plotted
# before and after the filtering to show the effect. The drop after 600 Hz
# appears because the data was filtered during the acquisition.
meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
raw.plot_psd(picks=meg_picks)
notches = np.arange(60, 181, 60)
raw.notch_filter(notches)
raw_erm.notch_filter(notches)
raw.plot_psd(picks=meg_picks)

##############################################################################
# We also lowpass filter the data at 100 Hz to remove the hf components.
raw.filter(None, 100.)
raw_erm.filter(None, 100.)

##############################################################################
# Epoching and averaging.
# First some parameters are defined and events extracted from the stimulus
# channel (UPPT001). The rejection thresholds are defined as peak-to-peak
# values and are defined in T / m for gradiometers, T for magnetometers and
# V for EOG and EEG channels.
tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)
# find events
events = mne.find_events(raw, stim_channel='UPPT001')

##############################################################################
# The event timing is adjusted by comparing the trigger times on detected
# sound onsets on channel UADC001-4408.
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

##############################################################################
# We mark a set of bad channels that seem noisier than others. This can also
# be done interactively with ``raw.plot`` by clicking the channel name
# (or the line). The marked channels are added as bad when the browser window
# is closed.
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

##############################################################################
# The epochs (trials) are created for MEG channels. Frist we find the picks
# for MEG and EOG channels. Then the epochs are constructed using these picks.
# The annotated bad segments are also used for removal of bad epochs by
# default. To turn off rejection by bad segments (as was done earlier with
# saccades) you can use keyword ``segment_reject=False``.
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False,
                    proj=True)

##############################################################################
# We only use first 40 good epochs from each run. Since some of the epochs
# were rejected, the epoch indices don't respond to the original anymore. By
# investigating the event timings, we conclude that the first epoch from the
# second run corresponds to epoch number 192.
selection = np.concatenate([range(40), range(192, 232)])
epochs_standard = epochs['standard'][selection]

##############################################################################
# The averages for each conditions are computed and plotted. In both
# conditions we can see the P50 and N100 responses. The mismatch negativity is
# visible only in the deviant condition around 100-200 ms. P200 is also
# visible around 170 ms in both conditions but much stronger in the standard
# condition. P300 is visible in deviant condition only (decision making in
# preparation of the button press).
evoked_standard = epochs_standard.average()
evoked_deviant = epochs['deviant'].average()

evoked_standard.plot(window_title='Standard', gfp=True)
evoked_deviant.plot(window_title='Deviant', gfp=True)

##############################################################################
# Show activations as topography figures.
times = np.arange(0, 0.351, 0.025)
evoked_standard.plot_topomap(times=times, title='Standard')
evoked_deviant.plot_topomap(times=times, title='Deviant')

##############################################################################
# We can see the MMN effect more clearly by looking at the difference between
# the two conditions. P50 and N100 are no longer visible, but MMN/P200 and
# P300 can be seen more clearly.
evoked_difference = evoked_standard.copy()
evoked_difference.data = evoked_deviant.data - evoked_standard.data
evoked_difference.plot(window_title='Difference', gfp=True)

##############################################################################
# Source estimation.
if run_source_space:

##############################################################################
# We compute the noise covariance matrix from the empty room measurement
# and use it for the other runs.
    reject = dict(mag=4e-12)
    cov = mne.compute_raw_covariance(raw_erm, reject=reject)
    cov.plot(raw_erm.info)

##############################################################################
# To save time, we read the source space from a file, but it can also be
# constructed with a function:
# src = mne.setup_source_space(subject, subjects_dir=subjects_dir)
# For more information: :ref:`CHDBBCEJ`, :class:`mne.setup_source_space`.
    src_fname = op.join(subjects_dir, 'bst_auditory', 'bem',
                        'bst_auditory-oct-6-src.fif')
    src = mne.read_source_spaces(src_fname)

##############################################################################
# The transformation is read from a file. To coregister the data manually, see
# :ref:`ch_interactive_analysis` or :func:`mne.gui.coregistration`.
    trans_fname = op.join(data_path, 'MEG', 'bst_auditory',
                          'bst_auditory-trans.fif')
    trans = mne.read_trans(trans_fname)

##############################################################################
# The head surfaces are read from a file. Since the data only contains MEG
# channels, we only need the inner skull surface for making the forward
# solution. For more information: :ref:`create_bem_model`,
# :func:`mne.bem.make_watershed_bem`.
    surfs_fname = op.join(subjects_dir, 'bst_auditory', 'bem',
                          'bst_auditory-inner_skull.fif')
    surfs = mne.read_bem_surfaces(surfs_fname)
    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(evoked_standard.info, trans=trans, src=src,
                                    bem=bem)

##############################################################################
# The sources are computed using dSPM method.
    inv = mne.minimum_norm.make_inverse_operator(evoked_standard.info, fwd,
                                                 cov)
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc_standard = mne.minimum_norm.apply_inverse(evoked_standard, inv,
                                                  lambda2, 'dSPM')
    stc_deviant = mne.minimum_norm.apply_inverse(evoked_deviant, inv, lambda2,
                                                 'dSPM')
    stc_difference = mne.minimum_norm.apply_inverse(evoked_difference, inv,
                                                    lambda2, 'dSPM')

##############################################################################
# The source estimates are plotted on an inflated brain surface. You can
# navigate in time by adjusting it on the time viewer.
    brain = stc_standard.plot(subjects_dir=subjects_dir, subject=subject,
                              surface='inflated', time_viewer=True,
                              hemi='both')
    brain.set_data_time_index(480)

    brain = stc_deviant.plot(subjects_dir=subjects_dir, subject=subject,
                             surface='inflated', time_viewer=True, hemi='both')
    brain.set_data_time_index(480)

    brain = stc_difference.plot(subjects_dir=subjects_dir, subject=subject,
                                surface='inflated', time_viewer=True,
                                hemi='both')
    brain.set_data_time_index(480)
