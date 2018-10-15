"""
=============================================
Plotting topographic arrowmaps of evoked data
=============================================

Load evoked data and plot arrowmaps along with the topomap for selected time
points. An arrowmap is based upon the Hosaka-Cohen transformation and represents
an estimation of the current flow underneath the MEG sensors. They are a poor man's MNE.

See [1]_ for details.

Reference
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
import mne
from mne.datasets import sample
from mne.datasets.brainstorm import bst_raw
from mne import read_evokeds
from mne.viz import plot_arrowmap

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked data
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))
evoked_mag = evoked.copy().pick_types(meg='mag')
evoked_grad = evoked.copy().pick_types(meg='grad')

# plot magnetometer data as an arrowmap along with the topoplot at the time
# of the maximum sensor space activity
max_time_idx = np.abs(evoked_mag.data).mean(axis=0).argmax()
plot_arrowmap(evoked_mag.data[:, max_time_idx], evoked_mag.info)

# data can be projected from info_from to info_to using mne mapping
# plot gradiometer data as an arrowmap along with the topoplot at the time
# of the maximum sensor space activity
plot_arrowmap(evoked_grad.data[:, max_time_idx], info_from=evoked_grad.info,
              info_to=evoked_mag.info)


# data can be projected from info_from (Vectorview 101) to
# high density info_to (CTF 272)
# info_to can be use to project data from in_from
# plot gradiometer data as an arrowmap along with the topoplot at the time
path = bst_raw.data_path()
raw_fname = path + '/MEG/bst_raw/' \
                   'subj001_somatosensory_20111109_01_AUX-f_raw.fif'
raw_ctf = mne.io.read_raw_fif(raw_fname, preload=True)
raw_ctf.pick_types(meg=True, ref_meg=False)
plot_arrowmap(evoked_grad.data[:, max_time_idx], info_from=evoked_grad.info,
              info_to=raw_ctf.info, scale=2e-10)