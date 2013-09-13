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
import matplotlib.pyplot as pl
#from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

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
event_id, tmin, tmax = 1, -0.2, 0.5
#events = mne.read_events(event_fname)[:3]  # TODO: Use all events
events = mne.read_events(event_fname)  # TODO: Use all events
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))
evoked = epochs.average()

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

###############################################################################
# New T-F Beamformer Code

# Just noting of reference values used in the Dalal paper for frequency bins
# (the high gamma band was further subdivided) and time window lengths
#freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz
#window_lenghts = [300, 200, 150]  # ms

# Setting frequency bins as in Dalal et al. 2008
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 300)]  # Hz
win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
#win_lengths = [0.2, 0.2, 0.2, 0.2]  # s

# Setting time windows, please note tmin stretches over the baseline, which is
# selected to be as long as the longest time window. This enables a smooth and
# accurate localization of activity in time
#tmin = 0.0  # s
#tstep = 0.05  # s
tmin = -0.2
tstep = 0.05

stcs = tf_dics(epochs, forward, label=label, tmin=tmin, tmax=0.5,
               tstep=tstep, win_lengths=win_lengths, control=(-0.2, 0.0),
               freq_bins=freq_bins)

# Gathering results for each time window
# TODO: Should be frequencies, but stcs for time windows are returned now
source_power = []
for stc in stcs:
    source_power.append(stc.data)

# Finding the source with maximum source power to plot spectrogram for that
# source
source_power = np.array(source_power)
max_index = np.unravel_index(source_power.argmax(), source_power.shape)
max_source = max_index[1]

# Preparing the time and frequency grid for plotting
time_bounds = [tmin]
for i in range(int(np.floor((epochs.times[-1] - tmin) / tstep))):
    time_bounds.append(tmin + (i + 1) * tstep)
freq_bounds = [freq_bins[0][0]]
for freq_bin in freq_bins:
    freq_bounds.append(freq_bin[1])
time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

# Plotting the results
# TODO: The gap between 55 and 65 Hz should be marked on the final spectrogram
pl.pcolor(time_grid, freq_grid, source_power[:, max_source, :],
          cmap=pl.cm.jet)
ax = pl.gca()
pl.xlabel('Time window boundaries [s]')
ax.set_xticks(time_bounds)
pl.xlim(time_bounds[0], time_bounds[-1])
pl.ylabel('Frequency bin boundaries [Hz]')
pl.yscale('log')
ax.set_yticks(freq_bounds)
ax.set_yticklabels([np.round(freq, 2) for freq in freq_bounds])
pl.ylim(freq_bounds[0], freq_bounds[-1])
pl.grid(True, ls='-')
pl.colorbar()
pl.show()






# Setting frequency bins so that each frequency up to a certain limit is
# plotted separately
#fmax = 59
#n_times = int(time_step * sfreq)
#freqs = fftfreq(n_times, 1. / sfreq)
#freqs = freqs[(freqs >= 0) & (freqs < fmax)]
#freq_bins = []
#freq_bounds = [np.mean([freqs[0], freqs[1]])]
##freq_ticks = []
#for i in range(len(freqs) - 2):
#    freq_bins.append((np.mean([freqs[i], freqs[i + 1]]),
#                      np.mean([freqs[i + 1], freqs[i + 2]])))
#    freq_bounds.append(np.mean([freqs[i + 1], freqs[i + 2]]))
#    #freq_ticks.append(np.round(freqs[i+1], 2))
