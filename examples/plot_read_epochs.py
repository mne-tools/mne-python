"""
==================================
Reading epochs from a raw FIF file
==================================

This script shows how to read the epochs from a raw file given
a list of events. For illustration, we compute the evoked responses
for both MEG and EEG data by averaging all the epochs.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import numpy as np

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
raw = fiff.setup_read_raw(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
include = [] # or stim channel ['STI 014']
exclude = raw['info']['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more

# MEG Magnetometers
meg_mag_picks = fiff.pick_types(raw['info'], meg='mag', eeg=False, stim=False,
                                            include=include, exclude=exclude)
meg_mag_data, times, channel_names = mne.read_epochs(raw, events, event_id,
                            tmin, tmax, picks=meg_mag_picks, baseline=(None, 0))
meg_mag_epochs = np.array([d['epoch'] for d in meg_mag_data]) # as 3D matrix
meg_mag_evoked_data = np.mean(meg_mag_epochs, axis=0) # compute evoked fields

# MEG
meg_grad_picks = fiff.pick_types(raw['info'], meg='grad', eeg=False,
                                stim=False, include=include, exclude=exclude)
meg_grad_data, times, channel_names = mne.read_epochs(raw, events, event_id,
                        tmin, tmax, picks=meg_grad_picks, baseline=(None, 0))
meg_grad_epochs = np.array([d['epoch'] for d in meg_grad_data]) # as 3D matrix
meg_grad_evoked_data = np.mean(meg_grad_epochs, axis=0) # compute evoked fields

# EEG
eeg_picks = fiff.pick_types(raw['info'], meg=False, eeg=True, stim=False,
                                            include=include, exclude=exclude)
eeg_data, times, channel_names = mne.read_epochs(raw, events, event_id,
                            tmin, tmax, picks=eeg_picks, baseline=(None, 0))
eeg_epochs = np.array([d['epoch'] for d in eeg_data]) # as 3D matrix
eeg_evoked_data = np.mean(eeg_epochs, axis=0) # compute evoked potentials

###############################################################################
# View evoked response
import pylab as pl
pl.clf()
pl.subplot(3, 1, 1)
pl.plot(times, meg_mag_evoked_data.T)
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (T)')
pl.title('MEG (Magnetometers) evoked field')
pl.subplot(3, 1, 2)
pl.plot(times, meg_grad_evoked_data.T)
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (T/m)')
pl.title('MEG (Gradiometers) evoked field')
pl.subplot(3, 1, 3)
pl.plot(times, eeg_evoked_data.T)
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Potential (V)')
pl.title('EEG evoked potential')
pl.subplots_adjust(0.175, 0.04, 0.94, 0.94, 0.2, 0.53)
pl.show()
