"""
=============================================
Plotting topographic arrowmaps of evoked data
=============================================

Load evoked data and plot arrowmaps along with the topomap for selected time
points. Arrowmap is based upon Hosaka-Cohen transformation and represents
actual current underneath the MEG sensors, they are poor man MNE

See [1]_ for details.

References
----------
.. [1] D. Cohen, H. Hosaka
   "Part II magnetic field produced by a current dipole",
    Journal of electrocardiology, Volume 9, Number 4, pp. 409-417, 1976.
    DOI: 10.1016/S0022-0736(76)80041-6
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
evoked_mag = evoked.copy().pick_types(meg='mag')
evoked_grad = evoked.copy().pick_types(meg='grad')

# plot magnetometer data as arrowmap along with topoplot at the time
# of the maximum sensor space activity
max_time_idx = np.abs(evoked_mag.data).mean(axis=0).argmax()
plot_arrowmap(evoked_mag.data[:, max_time_idx], evoked_mag.info)


# plot gradiometer data as arrowmap along with topoplot at the time
# of the maximum sensor space activity
max_time_idx = np.abs(evoked_grad.data).mean(axis=0).argmax()
plot_arrowmap(evoked_grad.data[:, max_time_idx], evoked_grad.info,
              info_to=evoked_mag.info)
