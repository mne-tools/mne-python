"""
=========================================
Compute source power using DICS beamfomer
=========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) filter from single trial
activity to estimate source power for two frequencies of interest.

The original reference for DICS is:
Gross et al. Dynamic imaging of coherent sources: Studying neural interactions
in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
"""

# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import mne

import numpy as np

from mne.fiff import Raw
from mne.datasets import sample
from mne.time_frequency import compute_epochs_csd
from mne.beamformer import dics_source_power

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name
subjects_dir = data_path + '/subjects'

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

# Computing the data and noise cross-spectral density matrices
# The time-frequency window was chosen on the basis of spectrograms from
# example time_frequency/plot_time_frequency.py
data_csd, freqs = compute_epochs_csd(epochs, mode='multitaper', tmin=0.04,
                                     tmax=0.15, fmin=30, fmax=50, fsum=False)
noise_csd, _ = compute_epochs_csd(epochs, mode='multitaper', tmin=-0.11,
                                  tmax=-0.001, fmin=30, fmax=50, fsum=False)

# Compute DICS spatial filter and estimate source time courses on evoked data
stc = dics_source_power(epochs.info, forward, noise_csd, data_csd, freqs)

# Plot source power separately for each frequency of interest
brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir,
                 time_label='frequency = %0.1f Hz')
for i, freq in enumerate(freqs):
    data = stc.data[:, i]
    amp_max = np.max(data)
    amp_min = np.mean([amp_max, min(data)])
    amp_mid = np.mean([amp_max, amp_min])
    brain.set_data_time_index(i)
    brain.scale_data_colormap(fmin=amp_min, fmid=amp_mid, fmax=amp_max,
                              transparent=True)
    brain.show_view('lateral')
    brain.save_image('DICS_source_power_freq_%d.png' % freq)
