from copy import copy

import numpy as np
import matplotlib.pyplot as pl
#from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

import mne

from mne.fiff import Raw
from mne.datasets import sample
from mne.beamformer import generate_filtered_epochs, tf_lcmv

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

# Read epochs
event_id, epoch_tmin, epoch_tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)[:3]  # TODO: Use all events
#events = mne.read_events(event_fname)  # TODO: Use all events

###############################################################################
# Time-frequency beamforming based on LCMV

# Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
#win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
win_lengths = [0.2, 0.2, 0.2, 0.2]  # s

# Setting time windows
tmin = -0.2
tmax = 0.5
tstep = 0.2
control = (-0.2, 0.0)

filtered_epochs = generate_filtered_epochs(freq_bins, 4, raw, events, event_id,
                                           epoch_tmin, epoch_tmax, control,
                                           picks=picks,
                                           reject=dict(grad=4000e-13,
                                                       mag=4e-12))

stcs = []
for i, epochs_band in enumerate(filtered_epochs):
    stc = tf_lcmv(epochs_band, forward, label=label, tmin=tmin, tmax=tmax,
                  tstep=tstep, win_length=win_lengths[i], control=control,
                  reg=0.05)
    stcs.append(stc)


# Gathering results for each time window
source_power = [stc.data for stc in stcs]

# Finding the source with maximum source power to plot spectrogram for that
# source
source_power = np.array(source_power)
max_source = np.unravel_index(source_power.argmax(), source_power.shape)[1]

# Preparing time-frequency cell boundaries for plotting
time_bounds = np.arange(tmin, tmax + 1 / raw.info['sfreq'], tstep)
freq_bounds = sorted(set(np.ravel(freq_bins)))
freq_ticks = copy(freq_bounds)

# If there is a gap in the frequency bins it will be covered with a gray bar
gap_bounds = []
for i in range(len(freq_bins) - 1):
    lower_bound = freq_bins[i][1]
    upper_bound = freq_bins[i+1][0]
    if lower_bound != upper_bound:
        freq_bounds.remove(lower_bound)
        gap_bounds.append((lower_bound, upper_bound))

# Preparing time-frequency grid for plotting
time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

# Plotting the results
pl.figure(figsize=(13, 9))
pl.pcolor(time_grid, freq_grid, source_power[:, max_source, :],
          cmap=pl.cm.jet)
pl.title('Source power in overlapping time-frequency windows calculated using '
         'LCMV')
ax = pl.gca()
pl.xlabel('Time window boundaries [s]')
ax.set_xticks(time_bounds)
pl.xlim(time_bounds[0], time_bounds[-1])
pl.ylabel('Frequency bin boundaries [Hz]')
pl.yscale('log')
ax.set_yticks(freq_ticks)
ax.set_yticklabels([np.round(freq, 2) for freq in freq_ticks])
pl.ylim(freq_bounds[0], freq_bounds[-1])
pl.grid(True, ls='-')
pl.colorbar()

# Horizontal bar across frequency gaps
for lower_bound, upper_bound in gap_bounds:
    pl.barh(lower_bound, time_bounds[-1] - time_bounds[0], upper_bound -
            lower_bound, time_bounds[0], color='lightgray')

pl.show()
