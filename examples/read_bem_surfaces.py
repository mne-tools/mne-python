"""
============================================
Reading BEM surfaces from a forward solution
============================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'

surfaces = mne.read_bem_surfaces(fname, add_geom=True)

print "Number of surfaces : %d" % len(surfaces)

###############################################################################
# Show result

head_col = (0.95, 0.83, 0.83)  # light pink
skull_col = (0.91, 0.89, 0.67)
brain_col = (0.67, 0.89, 0.91)  # light blue
colors = [head_col, skull_col, brain_col]

# 3D source space
from enthought.mayavi import mlab
mlab.clf()
for c, surf in zip(colors, surfaces):
    points = surf['rr']
    faces = surf['tris']
    mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], faces,
                         color=c, opacity=0.3)
