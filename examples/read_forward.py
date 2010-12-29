"""Reading a forward operator a.k.a. lead field matrix
"""
print __doc__

import fiff

fname = 'MNE-sample-data/subjects/sample/bem/sample-5120-bem-sol.fif'
fname = 'sm01a5-ave-oct-6-fwd.fif'

data = fiff.read_forward_solution(fname)
leadfield = data['sol']['data']

print "Leadfield size : %d x %d" % leadfield.shape

###############################################################################
# Show result

import pylab as pl
pl.matshow(leadfield[:306,:500])
pl.xlabel('sources')
pl.ylabel('sensors')
pl.title('Lead field matrix')
pl.show()

# 3D source space
lh_points = data['src'][0]['rr']
lh_faces = data['src'][0]['use_tris']
rh_points = data['src'][1]['rr']
rh_faces = data['src'][1]['use_tris']
from enthought.mayavi import mlab
mlab.triangular_mesh(lh_points[:,0], lh_points[:,1], lh_points[:,2], lh_faces)
mlab.triangular_mesh(rh_points[:,0], rh_points[:,1], rh_points[:,2], rh_faces)
