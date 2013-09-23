"""
=====================================
Time-frequency beamforming using LCMV
=====================================

Compute LCMV source power in a grid of time-frequency windows and display
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
from mne import compute_covariance
from mne.fiff import Raw
from mne.datasets import sample
from mne.epochs import generate_filtered_epochs
from mne.cov import regularize
from mne.beamformer import tf_lcmv
from mne.viz import plot_source_spectrogram


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

###############################################################################
# Read raw data, preload to allow filtering
raw = Raw(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read epochs without preloading to access the underlying raw object for
# filtering later in tf_lcmv
event_id, epoch_tmin, epoch_tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, epoch_tmin, epoch_tmax, proj=True,
                    picks=picks, baseline=(None, 0), reject=dict(grad=4000e-13,
                                                                 mag=4e-12))

###############################################################################
# Time-frequency beamforming based on LCMV

# Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
win_lengths = [0.2, 0.2, 0.2, 0.2]  # s

# Setting time windows
tmin = -0.2
tmax = 0.5
tstep = 0.2
baseline = (-0.2, 0.0)

n_jobs = 4
reg = 0.05

# Calculating noise covariance
# TODO: This should be done using empty room noise
filtered_epochs = generate_filtered_epochs(freq_bins, n_jobs, raw, events,
                                           event_id, epoch_tmin, epoch_tmax,
                                           baseline, picks=picks,
                                           reject=dict(grad=4000e-13,
                                                       mag=4e-12))
noise_covs = []
for epochs_band, win_length in zip(filtered_epochs, win_lengths):
    noise_cov = compute_covariance(epochs_band, tmin=tmin, tmax=tmin +
                                   win_length)
    noise_cov = regularize(noise_cov, epochs_band.info, mag=reg, grad=reg,
                           eeg=reg, proj=True)
    noise_covs.append(noise_cov)

# Computing LCMV solutions for time-frequency windows in a label in source
# space for faster computation, use label=None for full solution
stcs = tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
               freq_bins=freq_bins, reg=reg, label=label)

# Plotting source spectrogram for source with maximum activity
plot_source_spectrogram(stcs, freq_bins, source_index=None)
