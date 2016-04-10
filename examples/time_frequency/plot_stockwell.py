"""
=======================================================
Time frequency with Stockwell transform in sensor space
=======================================================

This script shows how to compute induced power and intertrial coherence
using the Stockwell transform, a.k.a. S-Transform.

"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.time_frequency import tfr_stockwell
from mne.datasets import somato

print(__doc__)

###############################################################################
# Set parameters
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -1., 3.

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
baseline = (None, 0)
events = mne.find_events(raw, stim_channel='STI 014')

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
                    preload=True)

###############################################################################
# Calculate power and intertrial coherence

epochs = epochs.pick_channels([epochs.ch_names[82]])  # reduce computation

power, itc = tfr_stockwell(epochs, fmin=6., fmax=30., decim=4, n_jobs=1,
                           width=.3, return_itc=True)

power.plot([0], baseline=None, mode=None, title='S-transform (power)')

itc.plot([0], baseline=None, mode=None, title='S-transform (ITC)')
