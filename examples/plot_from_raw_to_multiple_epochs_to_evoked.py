"""
====================================================================
Extract epochs for multiple conditions, save evoked response to disk
====================================================================

This script shows how to read the epochs for multiple conditions from
a raw file given a list of events. The epochs are averaged to produce
evoked data and then saved to disk.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne import io
from mne.datasets import sample
from mne.epochs import combine_event_ids
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_ids = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = io.Raw(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6))
# Let's equalize the trial counts in each condition
epochs.equalize_event_counts(['AudL', 'AudR', 'VisL', 'VisR'], copy=False)
# Now let's combine some conditions
combine_event_ids(epochs, ['AudL', 'AudR'], {'Auditory': 12}, copy=False)
combine_event_ids(epochs, ['VisL', 'VisR'], {'Visual': 34}, copy=False)

# average epochs and get Evoked datasets
evokeds = [epochs[cond].average() for cond in ['Auditory', 'Visual']]

# save evoked data to disk
mne.write_evokeds('sample_auditory_and_visual_eeg-ave.fif', evokeds)

###############################################################################
# View evoked response
import matplotlib.pyplot as plt
plt.clf()
ax = plt.subplot(2, 1, 1)
evokeds[0].plot(axes=ax)
plt.title('EEG evoked potential, auditory trials')
plt.ylabel('Potential (uV)')
ax = plt.subplot(2, 1, 2)
evokeds[1].plot(axes=ax)
plt.title('EEG evoked potential, visual trials')
plt.ylabel('Potential (uV)')
plt.show()
