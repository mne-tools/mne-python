"""
==================================
Shifting time-scale in evoked data
==================================

"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading evoked data
evoked = fiff.Evoked(fname, setno='Left Auditory',
                     baseline=(None, 0), proj=True)

picks = fiff.pick_channels(ch_names=evoked.info['ch_names'],
                           include="MEG 2332", exclude="bad")

# Create subplots
f, axarr = pl.subplots(3)
evoked.plot(exclude=[], picks=picks, axes=axarr[0],
            titles=dict(grad='Before time shifting'))

# Apply relative time-shift of 500 ms
evoked.shift_time(0.5, relative=True)

evoked.plot(exclude=[], picks=picks, axes=axarr[1],
            titles=dict(grad='Relative shift: 500 ms'))

# Apply absolute time-shift of 500 ms
evoked.shift_time(0.5, relative=False)

evoked.plot(exclude=[], picks=picks, axes=axarr[2],
            titles=dict(grad='Absolute shift: 500 ms'))
