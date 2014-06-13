"""
========================================
Plotting topographic maps of evoked data
========================================

Load evoked data and plot topomaps for selected time points.

"""
# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Tal Linzen <linzen@nyu.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
from mne.datasets import sample
from mne import read_evokeds

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

# plot magnetometer data as topomap at 1 time point : 100ms
evoked.plot_topomap(0.1, ch_type='mag', size=3, colorbar=False)

# set time instants in seconds (from 50 to 150ms in a step of 10ms)
times = np.arange(0.05, 0.15, 0.01)
# If times is set to None only 10 regularly spaced topographies will be shown

# plot magnetometer data as topomaps
evoked.plot_topomap(times, ch_type='mag')

# plot gradiometer data (plots the RMS for each pair of gradiometers)
evoked.plot_topomap(times, ch_type='grad')

# add channel labels and title
evoked.plot_topomap(0.1, ch_type='mag', show_names=True, colorbar=False,
                    size=8, title='Auditory response')
