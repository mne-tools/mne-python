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

from copy import copy

import numpy as np
import pylab as pl

import mne
from mne.fiff import Raw
from mne.datasets import sample
from mne.beamformer import tf_dics

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
stcs = tf_dics(epochs, forward, tmin, tmax, tstep, win_lengths, freq_bins,
               reg=0.001, label=label)

# Gathering results for each time window
source_power = [stc.data for stc in stcs]

# Finding the source with maximum source power
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
pl.figure(figsize=(7, 4))
pl.pcolor(time_grid, freq_grid, source_power[:, max_source, :],
          cmap=pl.cm.jet)
pl.title('TF source power calculated using DICS')
ax = pl.gca()
pl.xlabel('Time (s)')
ax.set_xticks(time_bounds)
pl.xlim(time_bounds[0], time_bounds[-1])
pl.ylabel('Frequency (Hz)')
pl.yscale('log')
ax.set_yticks(freq_ticks)
ax.set_yticklabels([np.round(freq, 2) for freq in freq_ticks])
pl.ylim(freq_bounds[0], freq_bounds[-1])
pl.grid(True, ls='-')
pl.colorbar()
mne.viz.tight_layout()

# Horizontal bar across frequency gaps
for lower_bound, upper_bound in gap_bounds:
    pl.barh(lower_bound, time_bounds[-1] - time_bounds[0], upper_bound -
            lower_bound, time_bounds[0], color='lightgray')

pl.show()
