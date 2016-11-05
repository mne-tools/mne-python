"""
============================================
Reading BEM surfaces from a forward solution
============================================

Plot BEM surfaces used for forward solution generation.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'

surfaces = mne.read_bem_surfaces(fname, patch_stats=True)

print("Number of surfaces : %d" % len(surfaces))

###############################################################################
# Show result
head_col = (0.95, 0.83, 0.83)  # light pink
skull_col = (0.91, 0.89, 0.67)
brain_col = (0.67, 0.89, 0.91)  # light blue
colors = [head_col, skull_col, brain_col]

# 3D source space
from mayavi import mlab  # noqa

mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
for c, surf in zip(colors, surfaces):
    points = surf['rr']
    faces = surf['tris']
    mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], faces,
                         color=c, opacity=0.3)
