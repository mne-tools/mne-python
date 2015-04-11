"""
=======================
Remap MEG channel types
=======================

In this example, MEG data are remapped from one
channel type to another. This is useful for combining statistics
for magnetometer and gradiometers. This process can be
computationally intensive.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>

# License: BSD (3-clause)

from mne.datasets import sample
from mne import read_evokeds
from mne.forward import compute_virtual_evoked

print(__doc__)

data_path = sample.data_path()
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

from_type, to_type, condition = 'grad', 'mag', 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))

virtual_evoked = compute_virtual_evoked(evoked, from_type=from_type,
                                        to_type=to_type)

virtual_evoked.plot_topomap(ch_type=to_type)
evoked.plot_topomap(ch_type=to_type)
