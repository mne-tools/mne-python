"""
==============================================
Reading a source space from a forward operator
==============================================

This example visualizes a source space mesh used by a forward operator.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import os.path as op

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')

patch_stats = True  # include high resolution source space
src = mne.read_source_spaces(fname, patch_stats=patch_stats)

# 3D source space (high sampling)
lh_points = src[0]['rr']
lh_faces = src[0]['tris']
rh_points = src[1]['rr']
rh_faces = src[1]['tris']

from mayavi import mlab  # noqa
mlab.figure(size=(600, 600), bgcolor=(0, 0, 0),)
mesh = mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                            lh_faces, colormap='RdBu')
mesh.module_manager.scalar_lut_manager.reverse_lut = True

mesh = mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                            rh_faces, colormap='RdBu')
mesh.module_manager.scalar_lut_manager.reverse_lut = True
