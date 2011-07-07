"""
========================================================
Extract epochs, average and save evoked response to disk
========================================================

This script shows how to read the epochs from a raw file given
a list of events. The epochs are averaged to produce evoked
data and then saved to disk.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path('.')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
exclude = raw.info['bads'] + ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                                            include=include, exclude=exclude)
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6))
evoked = epochs.average()  # average epochs and get an Evoked dataset.

evoked.save('sample_audvis_eeg-ave.fif')  # save evoked data to disk

###############################################################################
# View evoked response
times = 1e3 * epochs.times  # time in miliseconds
import pylab as pl
pl.clf()
pl.plot(times, 1e6 * evoked.data.T)
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Potential (uV)')
pl.title('EEG evoked potential')
pl.show()
