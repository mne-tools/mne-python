"""
===================================================
Compute induced power in the source space with dSPM
===================================================

Returns STC files ie source estimates of induced power
for different bands in the source space. The inverse method
is linear based on dSPM inverse operator.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
tmin, tmax, event_id = -0.2, 0.5, 1

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.find_events(raw)
inverse_operator = read_inverse_operator(fname_inv)

include = []
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                                stim=False, include=include, exclude=exclude)

# Load condition 1
event_id = 1
events = events[:10]  # take 10 events to keep the computation time low
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                    preload=True)

# Compute a source estimate per frequency band
bands = dict(alpha=[9, 11], beta=[18, 22])

stcs = source_band_induced_power(epochs, inverse_operator, bands, n_cycles=2,
                                 use_fft=False, n_jobs=1)

for b, stc in stcs.iteritems():
    stc.save('induced_power_%s' % b)

###############################################################################
# plot mean power
import pylab as pl
pl.plot(stcs['alpha'].times, stcs['alpha'].data.mean(axis=0), label='Alpha')
pl.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
pl.xlabel('Time (ms)')
pl.ylabel('Power')
pl.legend()
pl.title('Mean source induced power')
pl.show()
