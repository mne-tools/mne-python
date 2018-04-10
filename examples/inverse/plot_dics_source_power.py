"""
=========================================
Compute source power using DICS beamfomer
=========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) [1]_ filter from
single-trial activity to estimate source power across a frequency band.

References
----------
.. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
       interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#         Roman Goj <roman.goj@gmail.com>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import mne
from mne.datasets import sample
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'

###############################################################################
# Reading the raw data:
raw = mne.io.read_raw_fif(raw_fname)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Set picks
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

# Read epochs
event_id, tmin, tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))
evoked = epochs.average()

# Read forward operator
forward = mne.read_forward_solution(fname_fwd)

###############################################################################
# Computing the cross-spectral density matrix at 4 evenly spaced frequencies
# from 6 to 10 Hz. We use a decim value of 20 to speed up the computation in
# this example at the loss of accuracy.
csd = csd_morlet(epochs, tmin=0, tmax=0.5, decim=20,
                 frequencies=np.linspace(6, 10, 4))

# Compute DICS spatial filter and estimate source power.
filters = make_dics(epochs.info, forward, csd, reg=0.5)
stc, freqs = apply_dics_csd(csd, filters)

message = 'DICS source power in the 8-12 Hz frequency band'
brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir,
                 time_label=message)
