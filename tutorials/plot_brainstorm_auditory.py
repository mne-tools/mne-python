# -*- coding: utf-8 -*-
"""
====================================
Brainstorm auditory tutorial dataset
====================================

Here we compute the evoked from raw for the auditory Brainstorm
tutorial dataset. For comparison, see [1]_ and:

    http://neuroimage.usc.edu/brainstorm/Tutorials/Auditory

Experiment:

    - One subject, 2 acquisition runs 6 minutes each.
    - Each run contains 200 regular beeps and 40 easy deviant beeps.
    - Random ISI: between 0.7s and 1.7s seconds, uniformly distributed.
    - Button pressed when detecting a deviant with the right index finger.

The specifications of this dataset were discussed initially on the
`FieldTrip bug tracker <http://bugzilla.fcdonders.nl/show_bug.cgi?id=2300>`_.

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

import os.path as op
import pandas as pd
import numpy as np

import mne
from mne import combine_evoked
from mne.minimum_norm import apply_inverse
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.filter import notch_filter, filter_data

print(__doc__)

###############################################################################
# To reduce memory consumption and running time, some of the steps are
# precomputed. To run everything from scratch change this to False. With
# ``use_precomputed = False`` running time of this script can be several
# minutes even on a fast computer.
use_precomputed = True

###############################################################################
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

###############################################################################
# In the memory saving mode we use ``preload=False`` and use the memory
# efficient IO which loads the data on demand. However, filtering and some
# other functions require the data to be preloaded in the memory.
preload = not use_precomputed
raw = read_raw_ctf(raw_fname1, preload=preload)
n_times_run1 = raw.n_times
mne.io.concatenate_raws([raw, read_raw_ctf(raw_fname2, preload=preload)])
raw_erm = read_raw_ctf(erm_fname, preload=preload)

###############################################################################
# Data channel array consisted of 274 MEG axial gradiometers, 26 MEG reference
# sensors and 2 EEG electrodes (Cz and Pz).
# In addition:
#
#   - 1 stim channel for marking presentation times for the stimuli
#   - 1 audio channel for the sent signal
#   - 1 response channel for recording the button presses
#   - 1 ECG bipolar
#   - 2 EOG bipolar (vertical and horizontal)
#   - 12 head tracking channels
#   - 20 unused channels
#
# The head tracking channels and the unused channels are marked as misc
# channels. Here we define the EOG and ECG channels.
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
if not use_precomputed:
    # Leave out the two EEG channels for easier computation of forward.
    raw.pick_types(meg=True, eeg=False, stim=True, misc=True, eog=True,
                   ecg=True)

###############################################################################
# For noise reduction, a set of bad segments have been identified and stored
# in csv files. The bad segments are later used to reject epochs that overlap
# with them.
# The file for the second run also contains some saccades. The saccades are
# removed by using SSP. We use pandas to read the data from the csv files. You
# can also view the files with your favorite text editor.

annotations_df = pd.DataFrame()
offset = n_times_run1
for idx in [1, 2]:
    csv_fname = op.join(data_path, 'MEG', 'bst_auditory',
                        'events_bad_0%s.csv' % idx)
    df = pd.read_csv(csv_fname, header=None,
                     names=['onset', 'duration', 'id', 'label'])
    print('Events from run {0}:'.format(idx))
    print(df)

    df['onset'] += offset * (idx - 1)
    annotations_df = pd.concat([annotations_df, df], axis=0)

saccades_events = df[df['label'] == 'saccade'].values[:, :3].astype(int)

# Conversion from samples to times:
onsets = annotations_df['onset'].values / raw.info['sfreq']
durations = annotations_df['duration'].values / raw.info['sfreq']
descriptions = annotations_df['label'].values

annotations = mne.Annotations(onsets, durations, descriptions)
raw.annotations = annotations
del onsets, durations, descriptions

###############################################################################
# Here we compute the saccade and EOG projectors for magnetometers and add
# them to the raw data. The projectors are added to both runs.
saccade_epochs = mne.Epochs(raw, saccades_events, 1, 0., 0.5, preload=True,
                            reject_by_annotation=False)

projs_saccade = mne.compute_proj_epochs(saccade_epochs, n_mag=1, n_eeg=0,
                                        desc_prefix='saccade')
if use_precomputed:
    proj_fname = op.join(data_path, 'MEG', 'bst_auditory',
                         'bst_auditory-eog-proj.fif')
    projs_eog = mne.read_proj(proj_fname)[0]
else:
    projs_eog, _ = mne.preprocessing.compute_proj_eog(raw.load_data(),
                                                      n_mag=1, n_eeg=0)
raw.add_proj(projs_saccade)
raw.add_proj(projs_eog)
del saccade_epochs, saccades_events, projs_eog, projs_saccade  # To save memory

###############################################################################
# Visually inspect the effects of projections. Click on 'proj' button at the
# bottom right corner to toggle the projectors on/off. EOG events can be
# plotted by adding the event list as a keyword argument. As the bad segments
# and saccades were added as annotations to the raw data, they are plotted as
# well.
raw.plot(block=True)

###############################################################################
# Typical preprocessing step is the removal of power line artifact (50 Hz or
# 60 Hz). Here we notch filter the data at 60, 120 and 180 to remove the
# original 60 Hz artifact and the harmonics. The power spectra are plotted
# before and after the filtering to show the effect. The drop after 600 Hz
# appears because the data was filtered during the acquisition. In memory
# saving mode we do the filtering at evoked stage, which is not something you
# usually would do.
if not use_precomputed:
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False)
    raw.plot_psd(tmax=np.inf, picks=meg_picks)
    notches = np.arange(60, 181, 60)
    raw.notch_filter(notches)
    raw.plot_psd(tmax=np.inf, picks=meg_picks)

###############################################################################
# We also lowpass filter the data at 100 Hz to remove the hf components.
if not use_precomputed:
    raw.filter(None, 100., h_trans_bandwidth=0.5, filter_length='10s',
               phase='zero-double')

###############################################################################
# Epoching and averaging.
# First some parameters are defined and events extracted from the stimulus
# channel (UPPT001). The rejection thresholds are defined as peak-to-peak
# values and are in T / m for gradiometers, T for magnetometers and
# V for EOG and EEG channels.
tmin, tmax = -0.1, 0.5
event_id = dict(standard=1, deviant=2)
reject = dict(mag=4e-12, eog=250e-6)
# find events
events = mne.find_events(raw, stim_channel='UPPT001')

###############################################################################
# The event timing is adjusted by comparing the trigger times on detected
# sound onsets on channel UADC001-4408.
sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
min_diff = int(0.5 * raw.info['sfreq'])
diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
onsets = onsets[diffs > min_diff]
assert len(onsets) == len(events)
diffs = 1000. * (events[:, 0] - onsets) / raw.info['sfreq']
print('Trigger delay removed (μ ± σ): %0.1f ± %0.1f ms'
      % (np.mean(diffs), np.std(diffs)))
events[:, 0] = onsets
del sound_data, diffs

###############################################################################
# We mark a set of bad channels that seem noisier than others. This can also
# be done interactively with ``raw.plot`` by clicking the channel name
# (or the line). The marked channels are added as bad when the browser window
# is closed.
raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

###############################################################################
# The epochs (trials) are created for MEG channels. First we find the picks
# for MEG and EOG channels. Then the epochs are constructed using these picks.
# The epochs overlapping with annotated bad segments are also rejected by
# default. To turn off rejection by bad segments (as was done earlier with
# saccades) you can use keyword ``reject_by_annotation=False``.
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False,
                    proj=True)

###############################################################################
# We only use first 40 good epochs from each run. Since we first drop the bad
# epochs, the indices of the epochs are no longer same as in the original
# epochs collection. Investigation of the event timings reveals that first
# epoch from the second run corresponds to index 182.
epochs.drop_bad()
epochs_standard = mne.concatenate_epochs([epochs['standard'][range(40)],
                                          epochs['standard'][182:222]])
epochs_standard.load_data()  # Resampling to save memory.
epochs_standard.resample(600, npad='auto')
epochs_deviant = epochs['deviant'].load_data()
epochs_deviant.resample(600, npad='auto')
del epochs, picks

###############################################################################
# The averages for each conditions are computed.
evoked_std = epochs_standard.average()
evoked_dev = epochs_deviant.average()
del epochs_standard, epochs_deviant

###############################################################################
# Typical preprocessing step is the removal of power line artifact (50 Hz or
# 60 Hz). Here we notch filter the data at 60, 120 and 180 to remove the
# original 60 Hz artifact and the harmonics. Normally this would be done to
# raw data (with :func:`mne.io.Raw.filter`), but to reduce memory consumption
# of this tutorial, we do it at evoked stage.
if use_precomputed:
    sfreq = evoked_std.info['sfreq']
    notches = [60, 120, 180]
    for evoked in (evoked_std, evoked_dev):
        evoked.data[:] = notch_filter(evoked.data, sfreq, notches)
        evoked.data[:] = filter_data(evoked.data, sfreq, l_freq=None,
                                     h_freq=100.)

###############################################################################
# Here we plot the ERF of standard and deviant conditions. In both conditions
# we can see the P50 and N100 responses. The mismatch negativity is visible
# only in the deviant condition around 100-200 ms. P200 is also visible around
# 170 ms in both conditions but much stronger in the standard condition. P300
# is visible in deviant condition only (decision making in preparation of the
# button press). You can view the topographies from a certain time span by
# painting an area with clicking and holding the left mouse button.
evoked_std.plot(window_title='Standard', gfp=True)
evoked_dev.plot(window_title='Deviant', gfp=True)


###############################################################################
# Show activations as topography figures.
times = np.arange(0.05, 0.301, 0.025)
evoked_std.plot_topomap(times=times, title='Standard')
evoked_dev.plot_topomap(times=times, title='Deviant')

###############################################################################
# We can see the MMN effect more clearly by looking at the difference between
# the two conditions. P50 and N100 are no longer visible, but MMN/P200 and
# P300 are emphasised.
evoked_difference = combine_evoked([evoked_dev, -evoked_std], weights='equal')
evoked_difference.plot(window_title='Difference', gfp=True)

###############################################################################
# Source estimation.
# We compute the noise covariance matrix from the empty room measurement
# and use it for the other runs.
reject = dict(mag=4e-12)
cov = mne.compute_raw_covariance(raw_erm, reject=reject)
cov.plot(raw_erm.info)
del raw_erm

###############################################################################
# The transformation is read from a file. More information about coregistering
# the data, see :ref:`ch_interactive_analysis` or
# :func:`mne.gui.coregistration`.
trans_fname = op.join(data_path, 'MEG', 'bst_auditory',
                      'bst_auditory-trans.fif')
trans = mne.read_trans(trans_fname)

###############################################################################
# To save time and memory, the forward solution is read from a file. Set
# ``use_precomputed=False`` in the beginning of this script to build the
# forward solution from scratch. The head surfaces for constructing a BEM
# solution are read from a file. Since the data only contains MEG channels, we
# only need the inner skull surface for making the forward solution. For more
# information: :ref:`CHDBBCEJ`, :func:`mne.setup_source_space`,
# :ref:`create_bem_model`, :func:`mne.bem.make_watershed_bem`.
if use_precomputed:
    fwd_fname = op.join(data_path, 'MEG', 'bst_auditory',
                        'bst_auditory-meg-oct-6-fwd.fif')
    fwd = mne.read_forward_solution(fwd_fname)
else:
    src = mne.setup_source_space(subject, spacing='ico4',
                                 subjects_dir=subjects_dir, overwrite=True)
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3],
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(evoked_std.info, trans=trans, src=src,
                                    bem=bem)

inv = mne.minimum_norm.make_inverse_operator(evoked_std.info, fwd, cov)
snr = 3.0
lambda2 = 1.0 / snr ** 2
del fwd

###############################################################################
# The sources are computed using dSPM method and plotted on an inflated brain
# surface. For interactive controls over the image, use keyword
# ``time_viewer=True``.
# Standard condition.
stc_standard = mne.minimum_norm.apply_inverse(evoked_std, inv, lambda2, 'dSPM')
brain = stc_standard.plot(subjects_dir=subjects_dir, subject=subject,
                          surface='inflated', time_viewer=False, hemi='lh',
                          initial_time=0.1, time_unit='s')
del stc_standard, brain

###############################################################################
# Deviant condition.
stc_deviant = mne.minimum_norm.apply_inverse(evoked_dev, inv, lambda2, 'dSPM')
brain = stc_deviant.plot(subjects_dir=subjects_dir, subject=subject,
                         surface='inflated', time_viewer=False, hemi='lh',
                         initial_time=0.1, time_unit='s')
del stc_deviant, brain

###############################################################################
# Difference.
stc_difference = apply_inverse(evoked_difference, inv, lambda2, 'dSPM')
brain = stc_difference.plot(subjects_dir=subjects_dir, subject=subject,
                            surface='inflated', time_viewer=False, hemi='lh',
                            initial_time=0.15, time_unit='s')
