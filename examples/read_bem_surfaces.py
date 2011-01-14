"""Reading BEM surfaces
"""
print __doc__

from mne import fiff

fname = 'MNE-sample-data/subjects/sample/bem/sample-5120-bem-sol.fif'

surf = fiff.read_bem_surfaces(fname)

print "Number of surfaces : %d" % len(surf)

###############################################################################
# Show result

# 3D source space
points = surf[0]['rr']
faces = surf[0]['tris']
from enthought.mayavi import mlab
mlab.triangular_mesh(points[:,0], points[:,1], points[:,2], faces)
