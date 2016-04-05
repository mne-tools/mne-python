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
from mne.io import read_raw_ctf
from mne.filter import notch_filter, low_pass_filter

print(__doc__)


# To reduce memory consumption and running time, some of the steps are
# precomputed. To run everything from scratch change this to False.
use_precomputed = True

##############################################################################
# The data was collected with a CTF 275 system at 2400 Hz and low-pass
# filtered at 600 Hz. Here the data and empty room data files are read to
# construct instances of :class:`mne.io.Raw`.
data_path = bst_auditory.data_path()

subject = 'bst_auditory'
subjects_dir = op.join(data_path, 'subjects')

raw_fname1 = op.join(data_path, 'MEG', 'bst_auditory',
                     'S01_AEF_20131218_01.ds')
raw_fname2 = op.join(data_path, 'MEG', 'bst_auditory',
                     'S01_AEF_20131218_02.ds')
erm_fname = op.join(data_path, 'MEG', 'bst_auditory',
                    'S01_Noise_20131218_01.ds')
preload = not use_precomputed
raw1 = read_raw_ctf(raw_fname1, preload=preload)
raw2 = read_raw_ctf(raw_fname2, preload=preload)

raw_erm = read_raw_ctf(erm_fname, preload=preload)

raws = [raw1, raw2, raw_erm]
##############################################################################
# Data channel array consisted of 274 MEG axial gradiometers, 26 MEG reference
# sensors and 2 EEG electrodes (Cz and Pz).
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
for raw in raws[:2]:
    raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
    if not use_precomputed:
        # Leave out the two EEG channels for easier setup of source space.
        raw.pick_types(meg=True, eeg=False, stim=True, misc=True, eog=True,
                       ecg=True)

##############################################################################
# For noise reduction, a set of bad segments have been identified and stored
# in csv files. The bad segments are later used to reject epochs that overlap
# with them.
# The file for the second run also contains some saccades. The saccades are
# removed by using SSP. You can view the files with your favorite text editor.

for idx in [1, 2]:
    onsets = list()
    durations = list()
    descriptions = list()
    saccades = list()
    with open(data_path + '/MEG/bst_auditory/events_bad_0%s.csv' % idx,
              'r') as f:
        reader = csv.reader(f)
        for row in reader:
            onsets.append(int(row[0]))
            durations.append(int(row[1]))
            descriptions.append(row[3])
            if row[3] == 'saccade':  # Events for removal of saccades.
                saccades.append([int(row[0]), 0, 1])
    saccades = np.asarray(saccades)
    raw = raws[idx - 1]
    onsets = raw.index_as_time(onsets)  # Conversion from samples to times.
    durations = raw.index_as_time(durations)

    annot = mne.annotations.Annotations(onsets, durations, descriptions)
    raw.annotations = annot

##############################################################################
# Here we compute the saccade and EOG projectors for magnetometers and add
# them to the raw data. The projectors are added to both runs.
saccade_epochs = mne.Epochs(raw2, saccades, 1, 0., 0.5, preload=True,
                            segment_reject=False)

projs_saccade = mne.compute_proj_epochs(saccade_epochs, n_mag=1, n_eeg=0,
                                        desc_prefix='saccade')

cropped = raw2.crop(0.0, 200.0)  # Use only 200 s from 2. run to save memory.
projs_eog, eog_events = mne.preprocessing.compute_proj_eog(cropped.load_data(),
                                                           n_mag=1, n_eeg=0)
for raw in raws[:2]:
    raw.add_proj(projs_saccade)
    raw.add_proj(projs_eog)
del saccade_epochs, cropped  # To save memory

##############################################################################
# Visually inspect the effects of projections. Click on 'proj' button at the
# bottom right corner to toggle the projectors on/off. EOG events can be
# plotted by adding the event list as a keyword argument. As the bad segments
# and saccades were added as annotations to the raw data, they are plotted as
# well. You should also check that the EOG detection algorithm did it's job
# and the events are well aligned with the blinks.
raw1.plot()
raw2.plot(events=eog_events, block=True)

##############################################################################
# Typical preprocessing step is the removal of power line artifact (50 Hz or
# 60 Hz). Here we notch filter the data at 60, 120 and 180 to remove the
# original 60 Hz artifact and the harmonics. The power spectra are plotted
# before and after the filtering to show the effect. The drop after 600 Hz
# appears because the data was filtered during the acquisition. To save time
# and memory we do the filtering at evoked stage, which is not something you
# usually would do.
if not use_precomputed:
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
    raws[0].plot_psd(picks=meg_picks)
    notches = np.arange(60, 181, 60)
    for raw in raws:
        raw.notch_filter(notches)
    raws[0].plot_psd(picks=meg_picks)

##############################################################################
# We also lowpass filter the data at 100 Hz to remove the hf components.
if not use_precomputed:
    for raw in raws:
        raw.filter(None, 100.)

##############################################################################
# Epoching and averaging.
# First some parameters are defined and events extracted from the stimulus
# channel (UPPT001). The rejection thresholds are defined as peak-to-peak
# values and are in T / m for gradiometers, T for magnetometers and
# V for EOG and EEG channels.
tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)
# find events
events1 = mne.find_events(raw1, stim_channel='UPPT001')
events2 = mne.find_events(raw2, stim_channel='UPPT001')

##############################################################################
# The event timing is adjusted by comparing the trigger times on detected
# sound onsets on channel UADC001-4408.
events_list = [events1, events2]
for idx in [0, 1]:
    raw = raws[idx]
    events = events_list[idx]
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
for raw in raws[:2]:
    raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

##############################################################################
# The epochs (trials) are created for MEG channels. First we find the picks
# for MEG and EOG channels. Then the epochs are constructed using these picks.
# The annotated bad segments are also used for removal of bad epochs by
# default. To turn off rejection by bad segments (as was done earlier with
# saccades) you can use keyword ``segment_reject=False``.
picks = mne.pick_types(raw1.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

epochs1 = mne.Epochs(raw1, events1, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=False,
                     proj=True)
epochs2 = mne.Epochs(raw2, events2, event_id, tmin, tmax, picks=picks,
                     baseline=(None, 0), reject=reject, preload=False,
                     proj=True)

##############################################################################
# We only use first 40 good epochs from each run.
epochs_standard = mne.concatenate_epochs([epochs1['standard'][range(40)],
                                          epochs2['standard'][range(40)]])
epochs_deviant = mne.concatenate_epochs([epochs1['deviant'],
                                         epochs2['deviant']])


##############################################################################
# The averages for each conditions are computed.
evoked_std = epochs_standard.average()
evoked_dev = epochs_deviant.average()
del epochs_standard, epochs_deviant

##############################################################################
# Typical preprocessing step is the removal of power line artifact (50 Hz or
# 60 Hz). Here we notch filter the data at 60, 120 and 180 to remove the
# original 60 Hz artifact and the harmonics. Normally this would be done to
# raw data (with :func:`mne.io.Raw.filter`), but to reduce memory consumption
# of this tutorial, we do it at evoked stage.
if use_precomputed:
    sfreq = evoked_std.info['sfreq']
    nchan = evoked_std.info['nchan']
    notches = [60, 120, 180]
    for ch_idx in range(nchan):
        evoked_std.data[ch_idx] = notch_filter(evoked_std.data[ch_idx], sfreq,
                                               notches)
        evoked_dev.data[ch_idx] = notch_filter(evoked_dev.data[ch_idx], sfreq,
                                               notches)
        evoked_std.data[ch_idx] = low_pass_filter(evoked_std.data[ch_idx],
                                                  sfreq, 100)
        evoked_dev.data[ch_idx] = low_pass_filter(evoked_dev.data[ch_idx],
                                                  sfreq, 100)

##############################################################################
# Here we plot the ERF of standard and deviant conditions. In both conditions
# we can see the P50 and N100 responses. The mismatch negativity is visible
# only in the deviant condition around 100-200 ms. P200 is also visible around
# 170 ms in both conditions but much stronger in the standard condition. P300
# is visible in deviant condition only (decision making in preparation of the
# button press). You can view the topographies from a certain time span by
# painting an area with clicking and holding the left mouse button.
evoked_std.plot(window_title='Standard', gfp=True)
evoked_dev.plot(window_title='Deviant', gfp=True)

##############################################################################
# Show activations as topography figures.
times = np.arange(0, 0.351, 0.025)
evoked_std.plot_topomap(times=times, title='Standard')
evoked_dev.plot_topomap(times=times, title='Deviant')

##############################################################################
# We can see the MMN effect more clearly by looking at the difference between
# the two conditions. P50 and N100 are no longer visible, but MMN/P200 and
# P300 are emphasised.
evoked_difference = evoked_std.copy()
evoked_difference.data = evoked_dev.data - evoked_std.data
evoked_difference.plot(window_title='Difference', gfp=True)

##############################################################################
# Source estimation.
# We compute the noise covariance matrix from the empty room measurement
# and use it for the other runs.
reject = dict(mag=4e-12)
cov = mne.compute_raw_covariance(raw_erm, reject=reject)
cov.plot(raw_erm.info)

##############################################################################
# The transformation is read from a file. More information about coregistering
# the data, see :ref:`ch_interactive_analysis` or
# :func:`mne.gui.coregistration`.
trans_fname = op.join(data_path, 'MEG', 'bst_auditory',
                      'bst_auditory-trans.fif')
trans = mne.read_trans(trans_fname)

##############################################################################
# To save time and memory, the forward solution is read from a file. Set
# ``use_precomputed=False`` to build the forward solution from scratch.
# The head surfaces for constructing a BEM solution are read from a file.
# Since the data only contains MEG channels, we only need the inner skull
# surface for making the forward solution. For more information:
# :ref:`CHDBBCEJ`, :class:`mne.setup_source_space`, :ref:`create_bem_model`,
# :func:`mne.bem.make_watershed_bem`.
if use_precomputed:
    fwd_fname = op.join(data_path, 'MEG', 'bst_auditory',
                        'bst_auditory-meg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
else:
    src = mne.setup_source_space(subject, subjects_dir=subjects_dir,
                                 overwrite=True)
    surfs_fname = op.join(subjects_dir, 'bst_auditory', 'bem',
                          'bst_auditory-inner_skull.fif')
    surfs = mne.read_bem_surfaces(surfs_fname)

    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(evoked_std.info, trans=trans,
                                    src=src, bem=bem)

inv = mne.minimum_norm.make_inverse_operator(evoked_std.info, fwd, cov)
snr = 3.0
lambda2 = 1.0 / snr ** 2

##############################################################################
# The sources are computed using dSPM method and plotted on an inflated brain
# surface. For interactive controls over the image, use keyword
# ``time_viewer=True``.
# Standard condition.
stc_standard = mne.minimum_norm.apply_inverse(evoked_std, inv, lambda2, 'dSPM')
brain = stc_standard.plot(subjects_dir=subjects_dir, subject=subject,
                          surface='inflated', time_viewer=False, hemi='both',
                          views=['medial'])
brain.set_data_time_index(480)
del stc_standard, evoked_std
##############################################################################
# Deviant condition.
stc_deviant = mne.minimum_norm.apply_inverse(evoked_dev, inv, lambda2, 'dSPM')
brain = stc_deviant.plot(subjects_dir=subjects_dir, subject=subject,
                         surface='inflated', time_viewer=False, hemi='both',
                         views=['medial'])
brain.set_data_time_index(480)
del stc_deviant, evoked_dev
##############################################################################
# Difference.
stc_difference = mne.minimum_norm.apply_inverse(evoked_difference, inv,
                                                lambda2, 'dSPM')
brain = stc_difference.plot(subjects_dir=subjects_dir, subject=subject,
                            surface='inflated', time_viewer=False, hemi='both',
                            views=['medial'])
brain.set_data_time_index(600)
