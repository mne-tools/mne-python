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

#   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
include = [] # or stim channels ['STI 014']
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more

# EEG
eeg_picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False,
                                            include=include, exclude=exclude)
eeg_epochs = mne.Epochs(raw, events, event_id,
                            tmin, tmax, picks=eeg_picks, baseline=(None, 0))
eeg_evoked = eeg_epochs.average()
eeg_evoked_data = eeg_evoked.data

# MEG Magnetometers
meg_mag_picks = fiff.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                                            include=include, exclude=exclude)
meg_mag_epochs = mne.Epochs(raw, events, event_id,
                           tmin, tmax, picks=meg_mag_picks, baseline=(None, 0))
meg_mag_evoked = meg_mag_epochs.average()
meg_mag_evoked_data = meg_mag_evoked.data

# MEG
meg_grad_picks = fiff.pick_types(raw.info, meg='grad', eeg=False,
                                stim=False, include=include, exclude=exclude)
meg_grad_epochs = mne.Epochs(raw, events, event_id,
                        tmin, tmax, picks=meg_grad_picks, baseline=(None, 0))
meg_grad_evoked = meg_grad_epochs.average()
meg_grad_evoked_data = meg_grad_evoked.data

###############################################################################
# View evoked response
times = eeg_epochs.times
import pylab as pl
pl.clf()
pl.subplot(3, 1, 1)
pl.plot(1000*times, 1e13*meg_grad_evoked_data.T)
pl.ylim([-200, 200])
pl.xlim([1000*times[0], 1000*times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (fT/cm)')
pl.title('MEG (Gradiometers) evoked field')
pl.subplot(3, 1, 2)
pl.plot(1000*times, 1e15*meg_mag_evoked_data.T)
pl.ylim([-600, 600])
pl.xlim([1000*times[0], 1000*times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Magnetic Field (fT)')
pl.title('MEG (Magnetometers) evoked field')
pl.subplot(3, 1, 3)
pl.plot(1000*times, 1e6*eeg_evoked_data.T)
pl.xlim([1000*times[0], 1000*times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('Potential (uV)')
pl.title('EEG evoked potential')
pl.subplots_adjust(0.175, 0.07, 0.94, 0.94, 0.2, 0.53)
pl.show()
