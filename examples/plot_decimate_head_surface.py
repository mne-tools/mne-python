# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
"""
=======================
Decimating BEM surfaces
=======================

This can be useful to reduce computation time when
using a cloud of digitization points for coordinate alignment
instead of e.g. EEG-cap positions.

"""

print __doc__

import mne
from mne.surface import decimate_surface

path = mne.datasets.sample.data_path()
surf = mne.read_bem_surfaces(path + '/subjects/sample/bem/sample-head.fif')[0]
points = surf['rr']
triangles = surf['tris']

points_dec, faces_dec = decimate_surface(points, triangles, reduction=0.944)

try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab

head_col = (0.95, 0.83, 0.83)  # light pink

p, f = points_dec, faces_dec
mlab.triangular_mesh(p[:, 0], p[:, 1], p[:, 2], f, color=head_col)
