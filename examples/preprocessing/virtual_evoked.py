# -*- coding: utf-8 -*-
"""
.. _ex-virtual-evoked:

=======================
Remap MEG channel types
=======================

In this example, MEG data are remapped from one channel type to another.
This is useful to:

    - visualize combined magnetometers and gradiometers as magnetometers
      or gradiometers.
    - run statistics from both magnetometers and gradiometers while
      working with a single type of channels.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>

# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample

print(__doc__)

# read the evoked
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
fname = meg_path / 'sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory', baseline=(None, 0))

# %%
# First, let's call remap gradiometers to magnometers, and plot
# the original and remapped topomaps of the magnetometers.

# go from grad + mag to mag and plot original mag
virt_evoked = evoked.as_type('mag')
fig = evoked.plot_topomap(ch_type='mag')
fig.suptitle('mag (original)')
# %%

# plot interpolated grad + mag
fig = virt_evoked.plot_topomap(ch_type='mag')
fig.suptitle('mag (interpolated from mag + grad)')

# %%
# Now, we remap magnometers to gradiometers, and plot
# the original and remapped topomaps of the gradiometers

# go from grad + mag to grad and plot original grad
virt_evoked = evoked.as_type('grad')
fig = evoked.plot_topomap(ch_type='grad')
fig.suptitle('grad (original)')

# %%

# plot interpolated grad + mag
fig = virt_evoked.plot_topomap(ch_type='grad')
fig.suptitle('grad (interpolated from mag + grad)')
