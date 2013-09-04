"""
==========================
Time-frequency beamforming
==========================

The original reference is:
Dalal et al. Five-dimensional neuroimaging: Localization of the time-frequency
dynamics of cortical activity. NeuroImage (2008) vol. 40 (4) pp. 1686-1700
"""

# Author: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import logging
logger = logging.getLogger('mne')

import mne

from mne.fiff import Raw
from mne.datasets import sample
from mne.time_frequency import compute_epochs_csd
from mne.beamformer import dics_source_power

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'

###############################################################################
# Read raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read epochs
event_id, tmin, tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)[:3]  # TODO: Use all events
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))
evoked = epochs.average()

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)


###############################################################################
# New T-F Beamformer Code

# Just noting of reference values used in the Dalal paper for frequency bins
# (the high gamma band was further subdivided) and time window lengths
#freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz
#window_lenghts = [300, 200, 150]  # ms

# Setting window length and time step equal for a start
window_length = 0.2  # s
time_step = 0.2  # s
sfreq = epochs.info['sfreq']
time_step = np.round(time_step * sfreq) / sfreq
times = epochs.times
n_steps = int(np.floor((times[-1] - times[0]) / time_step))

# Setting frequency bins
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz

results = []

for i_time in range(n_steps):
    if i_time == 0:
        tmin = None
    else:
        tmin = tmax
    tmax = times[0] + (i_time + 1) * time_step

    # Calculating data and noise CSD matrices for current time window
    data_csds = []
    noise_csds = []
    for freq_bin in freq_bins:
        # TODO: Improve logging, which should be fairly informative, because
        # this is going to be taking a lot of time
        logger.info((i_time, freq_bin))

        data_csd = compute_epochs_csd(epochs, mode='fourier', tmin=tmin,
                                      tmax=tmax, fmin=freq_bin[0],
                                      fmax=freq_bin[1], fsum=True)
        data_csds.append(data_csd)

        # TODO: This is hacked for now to use always use the first time window
        # as noise, but noise normalization should be thought through (check
        # Dalal et al., they seem to describe it in detail)
        noise_csd = compute_epochs_csd(epochs, mode='fourier', tmin=None,
                                       tmax=times[0] + time_step,
                                       fmin=freq_bin[0], fmax=freq_bin[1],
                                       fsum=True)
        noise_csds.append(noise_csd)

    stc = dics_source_power(epochs.info, forward, noise_csds, data_csds)

    results.append(stc.data)
