"""
===================
Reading an STC file
===================

STC files contain activations on cortex ie. source
reconstructions
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import numpy as np
import mne
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis-meg-lh.stc'

stc = mne.read_stc(fname)

n_vertices, n_samples = stc['data'].shape
print "tmin : %s (s)" % stc['tmin']
print "tstep : %s" % stc['tstep']
print "tmax : %s (s)" % (stc['tmin'] + stc['tstep'] * n_samples)
print "stc data size: %s (nb of vertices) x %s (nb of samples)" % (
                                                    n_vertices, n_samples)

# View source activations
times = stc['tmin'] + stc['tstep'] * np.arange(n_samples)
import pylab as pl
pl.plot(times, stc['data'][::100,:].T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.show()
