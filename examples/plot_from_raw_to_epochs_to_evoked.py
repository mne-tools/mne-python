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
from mne.viz import plot_drop_log
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

#   Plot raw data
fig = raw.plot(events=events)

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True,
                        include=include, exclude='bads')
# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(eeg=80e-6, eog=150e-6))
evoked = epochs.average()  # average epochs and get an Evoked dataset.

evoked.save('sample_audvis_eeg-ave.fif')  # save evoked data to disk

###############################################################################
# View evoked response
times = 1e3 * epochs.times  # time in miliseconds
import pylab as pl
pl.figure()
evoked.plot()
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Potential (uV)')
pl.title('EEG evoked potential')
pl.show()

# Look at channels that caused dropped events, showing that the subject's
# blinks were likely to blame for most epochs being dropped
epochs.drop_bad_epochs()
plot_drop_log(epochs.drop_log, subject='sample')
