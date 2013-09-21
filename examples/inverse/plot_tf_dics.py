"""
=====================================
Time-frequency beamforming using DICS
=====================================

Compute DICS source power in a grid of time-frequency windows and display
results.

The original reference is:
Dalal et al. Five-dimensional neuroimaging: Localization of the time-frequency
dynamics of cortical activity. NeuroImage (2008) vol. 40 (4) pp. 1686-1700
"""

# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.fiff import Raw
from mne.datasets import sample
from mne.beamformer import tf_dics
from mne.viz import plot_source_spectrogram

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

###############################################################################
# Read raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read epochs
event_id, tmin, tmax = 1, -0.3, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

###############################################################################
# Time-frequency beamforming based on DICS

# Setting frequency bins as in Dalal et al. 2008
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz
win_lengths = [0.3, 0.2, 0.15, 0.1]  # s

# Setting time windows, please note tmin stretches over the baseline, which is
# selected to be as long as the longest time window. This enables a smooth and
# accurate localization of activity in time
tmin = -0.3  # s
tmax = 0.5  # s
tstep = 0.05  # s

# Solution constrained to label in source space for faster computation, use
# label=None for full solution
stcs = tf_dics(epochs, forward, tmin, tmax, tstep, win_lengths,
               baseline=(tmin, 0.0), freq_bins=freq_bins, reg=0.001,
               label=label)

# Plotting source spectrogram for source with maximum activity
plot_source_spectrogram(stcs, freq_bins, source_index=None)
