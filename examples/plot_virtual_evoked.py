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
from mne.datasets import somato

import matplotlib.pyplot as plt

print(__doc__)


def add_title(title):
    """Add nice titles."""
    plt.suptitle(title, verticalalignment='top', size='x-large')
    plt.gcf().set_size_inches(12, 2, forward=True)

# reject parameters and data paths
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'
event_id, tmin, tmax = 1, -0.2, 0.5
reject = dict(mag=4e-12, grad=4000e-13)

# setup for reading the raw data
raw = mne.io.Raw(raw_fname, proj=False)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject=reject,
                    preload=True, proj=False)
evoked = epochs.average()

# go from grad + mag to mag
virt_evoked = evoked.as_type('mag', copy=True)
evoked.plot_topomap(ch_type='mag')
add_title('mag (original)')
virt_evoked.plot_topomap(ch_type='mag')
add_title('mag (interpolated from mag + grad)')

# go from grad + mag to grad
virt_evoked = evoked.as_type('grad', copy=True)
evoked.plot_topomap(ch_type='grad')
add_title('grad (original)')
virt_evoked.plot_topomap(ch_type='grad')
add_title('grad (interpolated from mag + grad)')

plt.show()
