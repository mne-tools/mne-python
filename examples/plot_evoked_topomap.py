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
import matplotlib.pyplot as pl
import mne

path = mne.datasets.sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
evoked = mne.fiff.read_evoked(fname, 'Left Auditory', baseline=(None, 0))


times = np.arange(0.05, 0.15, 0.01)
vmax = 5e-13

mne.viz.plot_evoked_topomap(evoked, times, vmax=vmax, ch_type='mag')
