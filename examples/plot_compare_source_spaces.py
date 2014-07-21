# Author: Alan Leggitt <alan.leggitt@ucsf.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy.spatial import ConvexHull
from mayavi import mlab
from mne import (setup_source_space, setup_volume_source_space)
from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subj = 'sample'
aseg_fname = subjects_dir + '/sample/mri/aseg.mgz'
mri_fname = subjects_dir + '/sample/mri/brain.mgz'

# setup a cortical surface source space
surf = setup_source_space(subj, subjects_dir=subjects_dir, add_dist=False,
                          overwrite=True)

# setup a volume source space of the left cortical white matter
volume_label = 'Left-Cerebellum-Cortex'
sphere = (0, 0, 0, 120)
lh_ctx = setup_volume_source_space(subj, mri=aseg_fname, sphere=sphere,
                                   volume_label=volume_label,
                                   subjects_dir=subjects_dir)

#########################################
# Plot the positions of each source space

# left cortical surface
x1, y1, z1 = surf[0]['rr'].T

# left cerebellum cortex
x2, y2, z2 = lh_ctx[0]['rr'][lh_ctx[0]['inuse'].astype(bool)].T

# open a 3d figure
mlab.figure(1, bgcolor=(0, 0, 0))

# plot the left cortical surface
p1 = mlab.points3d(x1, y1, z1, color=(1, 0, 0))

# plot the convex hull bounding the left cerebellum
hull = ConvexHull(np.c_[x2, y2, z2])
mlab.triangular_mesh(x2, y2, z2, hull.simplices, color=(0.5, 0.5, 0.5),
                     opacity=0.3)

# plot the left cerebellum sources
mlab.points3d(x2, y2, z2, color=(1, 1, 0), scale_factor=0.001)

mlab.view(173.78, 101.75, 0.30, np.array([-0.03, -0.01,  0.03]))
mlab.roll(85)
mlab.show()
