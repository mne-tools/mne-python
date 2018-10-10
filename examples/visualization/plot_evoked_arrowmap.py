"""
=============================================
Plotting topographic arrowmaps of evoked data
=============================================

Load evoked data and plot arrowmaps along with
the topomap for selected time points.
Arrowmap is based upon Hosaka-Cohen transformation
and represents actual current
underneath the meg sensors, they are poor man MNE
"""
# Authors: Sheraz Khan <sheraz@khansheraz.com>
#
# License: BSD (3-clause)

import numpy as np
from mne.datasets import sample
from mne import read_evokeds
from mne.viz import plot_arrowmap

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

# plot magnetometer data as arrowmap along with topoplot at the time
# of the maximum sensor space activity
evoked.pick_types(meg='mag')
max_time_idx = np.abs(evoked.data).mean(axis=0).argmax()
plot_arrowmap(evoked.data[:, max_time_idx], evoked.info)
