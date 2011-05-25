"""
==============================================
Reading a source space from a forward operator
==============================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os.path as op

import mne
from mne.datasets import sample

data_path = sample.data_path('..')
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-eeg-oct-6p-fwd.fif')

add_geom = True  # include high resolution source space
src = mne.read_source_spaces(fname, add_geom=add_geom)

# 3D source space (high sampling)
lh_points = src[0]['rr']
lh_faces = src[0]['tris']
rh_points = src[1]['rr']
rh_faces = src[1]['tris']
from enthought.mayavi import mlab
mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                     lh_faces)
mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                     rh_faces)
