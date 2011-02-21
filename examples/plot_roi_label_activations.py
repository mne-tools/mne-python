"""
====================================================
Extracting the time series of activations in a label
====================================================


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
stc_fname = data_path + '/MEG/sample/sample_audvis-meg-lh.stc'
label_fname = data_path + '/subjects/sample/label/lh.BA1.label'

values, times, vertices = mne.label_time_courses(label_fname, stc_fname)

print "Number of vertices : %d" % len(vertices)

# View source activations
import pylab as pl
pl.plot(times, values.T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.show()
