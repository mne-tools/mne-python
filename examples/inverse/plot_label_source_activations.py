"""
====================================================
Extracting the time series of activations in a label
====================================================


"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path('..')
stc_fname = data_path + '/MEG/sample/sample_audvis-meg-lh.stc'
label = 'Aud-lh'
label_fname = data_path + '/MEG/sample/labels/%s.label' % label

values, times, vertices = mne.label_time_courses(label_fname, stc_fname)

print "Number of vertices : %d" % len(vertices)

# View source activations
import pylab as pl
pl.plot(1e3 * times, values.T)
pl.xlabel('time (ms)')
pl.ylabel('Source amplitude')
pl.title('Activations in Label : %s' % label)
pl.show()
