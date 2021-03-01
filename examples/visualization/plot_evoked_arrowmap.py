"""
=============================================
Plotting topographic arrowmaps of evoked data
=============================================

Load evoked data and plot arrowmaps along with the topomap for selected time
points. An arrowmap is based upon the Hosaka-Cohen transformation and
represents an estimation of the current flow underneath the MEG sensors.
They are a poor man's MNE.

See :footcite:`CohenHosaka1976` for details.

References
----------
.. footbibliography::
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

###############################################################################
# Plot magnetometer data as an arrowmap along with the topoplot at the time
# of the maximum sensor space activity:
max_time_idx = np.abs(evoked_mag.data).mean(axis=0).argmax()
plot_arrowmap(evoked_mag.data[:, max_time_idx], evoked_mag.info)

# Since planar gradiometers takes gradients along latitude and longitude,
# they need to be projected to the flatten manifold span by magnetometer
# or radial gradiometers before taking the gradients in the 2D Cartesian
# coordinate system for visualization on the 2D topoplot. You can use the
# ``info_from`` and ``info_to`` parameters to interpolate from
# gradiometer data to magnetometer data.

###############################################################################
# Plot gradiometer data as an arrowmap along with the topoplot at the time
# of the maximum sensor space activity:
plot_arrowmap(evoked_grad.data[:, max_time_idx], info_from=evoked_grad.info,
              info_to=evoked_mag.info)

###############################################################################
# Since Vectorview 102 system perform sparse spatial sampling of the magnetic
# field, data from the Vectorview (info_from) can be projected to the high
# density CTF 272 system (info_to) for visualization
#
# Plot gradiometer data as an arrowmap along with the topoplot at the time
# of the maximum sensor space activity:
path = bst_raw.data_path()
raw_fname = path + ('/MEG/bst_raw/'
                    'subj001_somatosensory_20111109_01_AUX-f.ds')
raw_ctf = mne.io.read_raw_ctf(raw_fname)
raw_ctf_info = mne.pick_info(
    raw_ctf.info, mne.pick_types(raw_ctf.info, meg=True, ref_meg=False))
plot_arrowmap(evoked_grad.data[:, max_time_idx], info_from=evoked_grad.info,
              info_to=raw_ctf_info, scale=6e-10)
