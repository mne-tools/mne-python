"""
=======================
Decimating BEM surfaces
=======================

This can be useful to reduce computation time when
using a cloud of digitization points for coordinate alignment
instead of e.g. EEG-cap positions.

"""
print __doc__
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.surface import decimate_surface

path = mne.datasets.sample.data_path()
surf = mne.read_bem_surfaces(path + '/subjects/sample/bem/sample-head.fif')[0]
points = surf['rr']
tris = surf['tris']

target_ntri = 30000  # reduce to 30000 meshes.
reduction = (target_ntri / surf['ntri'])
points_dec, tris_dec = decimate_surface(points, tris, reduction=reduction)

try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab

head_col = (0.95, 0.83, 0.83)  # light pink

p, f = points_dec, tris_dec
mlab.triangular_mesh(p[:, 0], p[:, 1], p[:, 2], f, color=head_col)
