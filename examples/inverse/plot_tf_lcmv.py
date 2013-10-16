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
from mne.event import make_fixed_length_events
from mne.beamformer import tf_lcmv
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
# Read raw data, preload to allow filtering
raw = Raw(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Pick a selection of magnetometer channels. A subset of all channels was used
# to speed up the example. For a solution based on all MEG channels use
# meg=True, selection=None and add grad=4000e-13 to the reject dictionary.
left_temporal_channels = mne.read_selection('Left-temporal')
picks = mne.fiff.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                            stim=False, exclude='bads',
                            selection=left_temporal_channels)
reject = dict(mag=4e-12)

# Read epochs. Note that preload is set to False to enable tf_lcmv to read the
# underlying raw object from epochs.raw, which would be set to None during
# preloading. Filtering is then performed on raw data in tf_lcmv and the epochs
# parameters passed here are used to create epochs from filtered data. However,
# reading epochs without preloading means that bad epoch rejection is delayed
# until later. To perform bad epoch rejection based on the reject parameter
# passed here, run epochs.drop_bad_epochs(). This is done automatically in
# tf_lcmv to reject bad epochs based on unfiltered data.
event_id, tmin, tmax = 1, -0.3, 0.5
events = mne.read_events(event_fname)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=False,
                    reject=reject)

# Read empty room noise, preload to allow filtering
raw_noise = Raw(noise_fname, preload=True)
raw_noise.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Create artificial events for empty room noise data
events_noise = make_fixed_length_events(raw_noise, event_id, duration=1.)
# Create an epochs object using preload=True to reject bad epochs based on
# unfiltered data
epochs_noise = mne.Epochs(raw_noise, events_noise, event_id, tmin, tmax,
                          proj=True, picks=picks, baseline=(None, 0),
                          preload=True, reject=reject)

# Make sure the number of noise epochs is the same as data epochs
epochs_noise = epochs_noise[:len(epochs.events)]

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

###############################################################################
# Time-frequency beamforming based on LCMV

# Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
win_lengths = [0.3, 0.2, 0.15, 0.1]  # s

# Setting the time step
tstep = 0.05

# Setting the noise covariance and whitened data covariance regularization
# parameters
noise_reg = 0.03
data_reg = 0.001

# Subtract evoked response prior to computation?
subtract_evoked = False

# Calculating covariance from empty room noise. To use baseline data as noise
# substitute raw for raw_noise, epochs for epochs_noise, and 0 for tmax.
# Note, if using baseline data, the averaged evoked response in the baseline 
# epoch should be flat.
noise_covs = []
for (l_freq, h_freq) in freq_bins:
    raw_band = raw_noise.copy()
    raw_band.filter(l_freq, h_freq, picks=epochs.picks, method='iir', n_jobs=1)
    epochs_band = mne.Epochs(raw_band, epochs_noise.events, event_id,
                             tmin=tmin, tmax=tmax, picks=epochs.picks,
                             proj=True)
                             
    noise_cov = compute_covariance(epochs_band)
    noise_cov = mne.cov.regularize(noise_cov, epochs_band.info, mag=noise_reg,
                                   grad=noise_reg, eeg=noise_reg, proj=True)
    noise_covs.append(noise_cov)
    del raw_band  # to save memory

# Computing LCMV solutions for time-frequency windows in a label in source
# space for faster computation, use label=None for full solution
stcs = tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
               freq_bins=freq_bins, subtract_evoked=subtract_evoked, 
               reg=data_reg, label=label)

# Plotting source spectrogram for source with maximum activity
plot_source_spectrogram(stcs, freq_bins, source_index=None, colorbar=True)
