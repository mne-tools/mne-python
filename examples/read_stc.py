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

import mne

fname = 'MNE-sample-data/MEG/sample/sample_audvis-meg-lh.stc'

stc = mne.read_stc(fname)

n_vertices, n_samples = stc['data'].shape
print "tmin : %s (s)" % stc['tmin']
print "tstep : %s" % stc['tstep']
print "tmax : %s (s)" % (stc['tmin'] + stc['tstep'] * n_samples)
print "stc data size: %s (nb of vertices) x %s (nb of samples)" % (
                                                    n_vertices, n_samples)
