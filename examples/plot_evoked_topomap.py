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


pl.figure(figsize=(12, 2))
times = np.arange(0.05, 0.15, 0.01)
vmax = 5e-13  # keep the colormap constant across plots
colorbar = False
for i, t in enumerate(times, 1):  # loop through time points
    pl.subplot(1, len(times), i)
    mne.viz.plot_evoked_topomap(evoked, t, vmax=vmax, ch_type='mag', colorbar=False)
    pl.title('%i ms' % (t * 1000))

# add a colorbar
pl.subplots_adjust(bottom=0.05, left=0.025, right=0.9, top=0.9, hspace=.5)
cax = pl.axes([0.925, 0.15, 0.01, 0.5])
pl.colorbar(cax=cax, ticks=[-vmax, 0, vmax])

pl.show()
