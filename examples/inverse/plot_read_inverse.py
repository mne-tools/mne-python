"""
=======================================================
Reading an inverse operator and view source space in 3D
=======================================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

data_path = sample.data_path('.')
fname = data_path
fname += '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'

inv = read_inverse_operator(fname)

print "Method: %s" % inv['methods']
print "fMRI prior: %s" % inv['fmri_prior']
print "Number of sources: %s" % inv['nsource']
print "Number of channels: %s" % inv['nchan']

###############################################################################
# Show result on 3D source space
lh_points = inv['src'][0]['rr']
lh_faces = inv['src'][0]['use_tris']
rh_points = inv['src'][1]['rr']
rh_faces = inv['src'][1]['use_tris']
from enthought.mayavi import mlab
mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                     lh_faces)
mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                     rh_faces)
