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

import numpy as np

import mne
from mne.fiff import Raw
from mne.event import make_fixed_length_events
from mne.datasets import sample
from mne.time_frequency import compute_epochs_csd
from mne.beamformer import tf_dics
from mne.viz import plot_source_spectrogram

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
noise_fname = data_path + '/MEG/sample/ernoise_raw.fif'
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
event_id, epoch_tmin, epoch_tmax = 1, -0.3, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, epoch_tmin, epoch_tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))

# Read empty room noise raw data
raw_noise = Raw(noise_fname)
raw_noise.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Create noise epochs and make sure the number of noise epochs corresponds to
# the number of data epochs
events_noise = make_fixed_length_events(raw_noise, event_id)
epochs_noise = mne.Epochs(raw, events_noise, event_id, epoch_tmin, epoch_tmax,
                          proj=True, picks=picks, baseline=(None, 0),
                          preload=True, reject=dict(grad=4000e-13, mag=4e-12))
epochs_noise.drop_epochs(range(len(epochs_noise.events) - len(epochs.events)))

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

###############################################################################
# Time-frequency beamforming based on DICS

# Setting frequency bins as in Dalal et al. 2008
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz
win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
n_ffts = [int(np.ceil(win_length * epochs.info['sfreq'])) for win_length in
          win_lengths]

# Setting time windows, please note tmin stretches over the baseline, which is
# selected to be as long as the longest time window. This enables a smooth and
# accurate localization of activity in time
tmin = -0.3  # s
tmax = 0.5  # s
tstep = 0.05  # s

# Calculating noise cross-spectral density from empty room noise for each
# frequency bin and the corresponding time window length. To calculate noise
# from the baseline period in the data, change epochs_noise to epochs
noise_csds = []
for freq_bin, win_length, n_fft in zip(freq_bins, win_lengths, n_ffts):
    noise_csd = compute_epochs_csd(epochs_noise, mode='fourier',
                                   fmin=freq_bin[0], fmax=freq_bin[1],
                                   fsum=True, tmin=tmin,
                                   tmax=tmin + win_length, n_fft=n_fft)
    noise_csd.data /= win_length
    noise_csds.append(noise_csd)

# Computing DICS solutions for time-frequency windows in a label in source
# space for faster computation, use label=None for full solution
stcs = tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
               freq_bins=freq_bins, n_ffts=n_ffts, reg=0.001, label=label)

# Plotting source spectrogram for source with maximum activity
plot_source_spectrogram(stcs, freq_bins, source_index=None)
