"""
==================================
Shifting time-scale in evoked data
==================================

"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

print __doc__

from mne import fiff
from mne.datasets import sample
from mne.viz import plot_evoked
import pylab as pl

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading evoked data
evoked = fiff.Evoked(fname, setno='Left Auditory',
                     baseline=(None, 0), proj=True)

picks = fiff.pick_channels(ch_names=evoked.info['ch_names'],
                           include="MEG 2332", exclude="bad")

# Create subplots
f, axarr = pl.subplots(3)
plot_evoked(evoked, exclude=[], picks=picks, axes=axarr[0],
            titles=dict(grad='Before time shifting'))

# Apply relative time-shift of 500 ms
evoked.time_shift(0.5, relative=True)

plot_evoked(evoked, exclude=[], picks=picks, axes=axarr[1],
            titles=dict(grad='Relative shift: 500 ms'))

# Apply absolute time-shift of 500 ms
evoked.time_shift(0.5, relative=False)

plot_evoked(evoked, exclude=[], picks=picks, axes=axarr[2],
            titles=dict(grad='Absolute shift: 500 ms'))