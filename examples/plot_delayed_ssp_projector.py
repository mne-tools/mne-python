"""
=========================================
Create evoked objects in delayed SSP mode
=========================================

This script shows how to apply SSP projectors at the evoked
stage. We first will extract Epochs and create evoked objects
with the required settings and then explore the different SSP
projectors at the evoked stage.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

#   Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

# pick Magnetometer channels
picks = fiff.pick_types(raw.info, meg='mag', stim=False, eog=True,
                        include=[], exclude='bads')

# Read epochs with proj == False
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12),
                    proj=False)

evoked = epochs.average()  # average epochs and get an Evoked dataset.


###############################################################################
# View evoked response with projectos idle
times = 1e3 * epochs.times  # time in milliseconds
import pylab as pl
pl.figure()
evoked.plot()
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('MEG evoked fields (fT)')
pl.title('Magnetometers | SSP off')
pl.show()

# now with projectors activated
pl.figure()
evoked.copy().apply_proj().plot()
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('MEG evoked fields (fT)')
pl.title('Magnetometers | SSP on')
pl.show()
