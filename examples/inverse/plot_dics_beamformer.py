"""
=====================================
Compute DICS beamfomer on evoked data
=====================================

Compute a Dynamic Imaging of Coherent Sources (DICS) beamformer from single
trial activity in a time-frequency window to estimate source time courses based
on evoked data.

The original reference for DICS is:
Gross et al. Dynamic imaging of coherent sources: Studying neural interactions
in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
"""
# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import mne

import matplotlib.pyplot as plt
import numpy as np

from mne.datasets import sample
from mne.time_frequency import csd_epochs
from mne.beamformer import dics

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name
subjects_dir = data_path + '/subjects'

###############################################################################
# Read raw data
raw = mne.io.read_raw_fif(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

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
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Computing the data and noise cross-spectral density matrices
# The time-frequency window was chosen on the basis of spectrograms from
# example time_frequency/plot_time_frequency.py
data_csd = csd_epochs(epochs, mode='multitaper', tmin=0.04, tmax=0.15,
                      fmin=6, fmax=10)
noise_csd = csd_epochs(epochs, mode='multitaper', tmin=-0.11, tmax=0.0,
                       fmin=6, fmax=10)

evoked = epochs.average()

# Compute DICS spatial filter and estimate source time courses on evoked data
stc = dics(evoked, forward, noise_csd, data_csd)

plt.figure()
ts_show = -30  # show the 40 largest responses
plt.plot(1e3 * stc.times,
         stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
plt.xlabel('Time (ms)')
plt.ylabel('DICS value')
plt.title('DICS time course of the 30 largest sources.')
plt.show()

# Plot brain in 3D with PySurfer if available
brain = stc.plot(hemi='rh', subjects_dir=subjects_dir,
                 initial_time=0.1, time_unit='s')
brain.show_view('lateral')

# Uncomment to save image
# brain.save_image('DICS_map.png')
