"""
========================================
Plotting topographic arrowmaps of evoked data
========================================

Load evoked data and plot topomaps for selected time points.
"""
# Authors: Sheraz Khan <sheraz@khansheraz.com>
#
# License: BSD (3-clause)


from mne.datasets import sample
from mne import read_evokeds
from mne.viz.topomap import plot_arrowmap

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

# plot magnetometer data as arrowmap at the maximum time
evoked.pick_types(meg='mag')
max_time_idx = evoked.data.__abs__().mean(axis=0).argmax()
plot_arrowmap(evoked.data[: ,max_time_idx], evoked.info)

