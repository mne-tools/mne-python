"""
===================================================
Reading a forward operator a.k.a. lead field matrix
===================================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'

fwd = mne.read_forward_solution(fname, surf_ori=True)
leadfield = fwd['sol']['data']

print "Leadfield size : %d x %d" % leadfield.shape

###############################################################################
# Show result

import pylab as pl
pl.matshow(leadfield[:, :500])
pl.xlabel('sources')
pl.ylabel('sensors')
pl.title('Lead field matrix')
pl.show()

# 3D source space
lh_points = fwd['src'][0]['rr']
lh_faces = fwd['src'][0]['use_tris']
rh_points = fwd['src'][1]['rr']
rh_faces = fwd['src'][1]['use_tris']
from enthought.mayavi import mlab
mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                     lh_faces)
mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                     rh_faces)
