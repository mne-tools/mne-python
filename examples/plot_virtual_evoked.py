"""
=======================
Remap MEG channel types
=======================

In this example, MEG data are remapped from one
channel type to another. This is useful for combining statistics
for magnetometer and gradiometers.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>

# License: BSD (3-clause)

import mne
from mne.datasets import sample

import matplotlib.pyplot as plt

print(__doc__)


def add_title(title):
    """Add nice titles."""
    plt.suptitle(title, verticalalignment='top', size='x-large')
    plt.gcf().set_size_inches(12, 2, forward=True)

# read the evoked
data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory', baseline=(None, 0))

# go from grad + mag to mag
virt_evoked = evoked.as_type('mag')
evoked.plot_topomap(ch_type='mag')
add_title('mag (original)')
virt_evoked.plot_topomap(ch_type='mag')
add_title('mag (interpolated from mag + grad)')

# go from grad + mag to grad
virt_evoked = evoked.as_type('grad')
evoked.plot_topomap(ch_type='grad')
add_title('grad (original)')
virt_evoked.plot_topomap(ch_type='grad')
add_title('grad (interpolated from mag + grad)')

plt.show()
