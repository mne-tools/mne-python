"""
========================================
Plotting topographic maps of evoked data
========================================

Load evoked data and plot topomaps for selected time points.

"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import mne

path = mne.datasets.sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
evoked = mne.fiff.read_evoked(fname, 'Left Auditory', baseline=(None, 0))


# set time instants in seconds (from 50 to 150ms in a step of 10ms)
times = np.arange(0.05, 0.15, 0.01)
# If times is set to None only 10 regularly spaced topographies will be shown

vmax = 5e-13  # set max value to scale colormap

# mne.viz.plot_evoked_topomap(evoked, times, vmax=vmax, ch_type='mag')
mne.viz.plot_evoked_topomap(evoked, times=times, vmax=vmax, ch_type='mag')
