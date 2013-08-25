# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.surface import decimate_surface

path = mne.datasets.sample.data_path()
surf = mne.read_bem_surfaces(path + '/subjects/sample/bem/sample-head.fif')
points = surf[0]['rr']

points_dec, faces_dec = decimate_surface(points, reduction=0.5)

# viz to check
# 3D source space
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab

head_col = (0.95, 0.83, 0.83)  # light pink

p, f = points_dec, faces_dec
mlab.triangular_mesh(p[:, 0], p[:, 1], p[:, 2], f, color=head_col)

