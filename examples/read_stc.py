"""Reading a raw file segment
"""
print __doc__

import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis-ave-7-meg-lh.stc'

stc = fiff.read_stc(fname)
fiff.write_stc("tmp.stc", stc)
stc2 = fiff.read_stc("tmp.stc")

from scipy import linalg
print linalg.norm(stc['data'] - stc2['data'])

# n_vertices, n_samples = stc['data'].shape
# print "tmin : %s (s)" % stc['tmin']
# print "tstep : %s" % stc['tstep']
# print "tmax : %s (s)" % (stc['tmin'] + stc['tstep'] * n_samples)
# print "stc data size: %s (nb of vertices) x %s (nb of samples)" % (
#                                                     n_vertices, n_samples)
# 
