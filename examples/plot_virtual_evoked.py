"""
=======================
Remap MEG channel types
=======================

In this example, MEG data are remapped from one
channel type to another. This is useful to:

    - visualize combined magnetometers and gradiometers as magnetometers
      or gradiometers.
    - run statistics from both magnetometers and gradiometers while
      working with a single type of channels.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>

# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

# read the evoked
data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory', baseline=(None, 0))

# go from grad + mag to mag
virt_evoked = evoked.as_type('mag')
evoked.plot_topomap(ch_type='mag', title='mag (original)')
virt_evoked.plot_topomap(ch_type='mag',
                         title='mag (interpolated from mag + grad)')

# go from grad + mag to grad
virt_evoked = evoked.as_type('grad')
evoked.plot_topomap(ch_type='grad', title='grad (original)')
virt_evoked.plot_topomap(ch_type='grad',
                         title='grad (interpolated from mag + grad)')
