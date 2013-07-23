"""
====================================================
Dynamic imaging of coherent sources (DICS) pSPM maps
====================================================

Work in progress.

"""

# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import mne

from mne.fiff import Raw
from mne.datasets import sample
from mne.time_frequency import compute_csd
from mne.beamformer import dics_epochs

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

###############################################################################
# Read raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read epochs
event_id, tmin, tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))
evoked = epochs.average()

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# TODO: Time and frequency windows should be selected on the basis of e.g. a
# spectrogram

# Computing the data and noise cross-spectral density matrices
data_csd = compute_csd(epochs, mode='multitaper', tmin=0.04, tmax=0.15, fmin=8,
                       fmax=12)
noise_csd = compute_csd(epochs, mode='multitaper', tmin=-0.11, tmax=0.0,
                        fmin=8, fmax=12)

stcs = dics_epochs(epochs, forward, noise_csd, data_csd)
