"""
=============================================
Joint topomap and time series plot of ERF/ERP
=============================================

Load evoked data and plot.

"""
# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from mne.datasets import sample
from mne import read_evokeds

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition,
                      baseline=(None, 0)).pick_types(meg='grad', eeg=False)

# Plot the evoked response with spatially color coded lines,
# topomaps for specified times, and the Global Field Power, but no sensors
ts_args = dict(gfp=True)
topomap_args = dict(sensors=False)
evoked.plot_joint(title=condition, times=[.07, .105],
                  ts_args=ts_args, topomap_args=topomap_args)

plt.show()
