"""
=================================
Warp head shapes between subjects
=================================

In this example, we warp data from one subject (sample) to another (fsaverage)
using a spherical harmonic approximation of surfaces, followed by thin-plate
spline (TPS) warping of the surface coordinates, as described in [1]_.

References
----------
.. [1] Darvas F, Ermer JJ, Mosher JC, Leahy RM (2006). "Generic head models for
       atlas-based EEG source analysis." Human Brain Mapping 27:129-143
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne

fsaverage_path = op.join(op.dirname(mne.__file__), 'data', 'fsaverage')
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

###############################################################################
# Load the source digitization, destination surfaces, and source surfaces
# (with everything in head coordinates):

# Digitization
info = mne.io.read_info(op.join(data_path, 'MEG', 'sample',
                                'sample_audvis_raw.fif'))
hsp = mne.bem.get_fitting_dig(info, ('cardinal', 'extra'))

# Destination head surface
fsaverage_surfs = [mne.read_bem_surfaces(op.join(fsaverage_path,
                                                 'fsaverage-%s.fif' % kind))[0]
                   for kind in ('head', 'inner_skull-bem')]
fsaverage_trans = mne.read_trans(op.join(fsaverage_path,
                                         'fsaverage-trans.fif'))
for surf in fsaverage_surfs:
    mne.surface.transform_surface_to(surf, 'head', fsaverage_trans, copy=False)

# Some source surfaces to transform as examples
sample_surfs = mne.read_bem_surfaces(
    op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem.fif'))
sample_trans = mne.read_trans(op.join(data_path, 'MEG', 'sample',
                                      'sample_audvis_raw-trans.fif'))
for surf in sample_surfs:
    mne.surface.transform_surface_to(surf, 'head', sample_trans, copy=False)

###############################################################################
# Transform surfaces using TPS warping:

warp = mne.transforms.SphericalSurfaceWarp()
warp.fit(source=hsp, destination=fsaverage_surfs[0]['rr'])
for surf in sample_surfs:
    surf['rr'] = warp.transform(surf['rr'])
    # recompute our normals (only used for viz here)
    mne.surface.complete_surface_info(surf, copy=False)
hsp = warp.transform(hsp)

###############################################################################
# Plot the transformed surfaces and digitization (blue) on template (black):

from mayavi import mlab  # noqa

t_color = (0.1, 0.3, 1)
fig = mlab.figure(size=(400, 600), bgcolor=(1., 1., 1.))
for surf, color in zip(fsaverage_surfs + sample_surfs,
                       [(0., 0., 0.)] * len(fsaverage_surfs) +
                       [t_color] * len(sample_surfs)):
    mesh = mlab.pipeline.triangular_mesh_source(
        *surf['rr'].T, triangles=surf['tris'])
    mesh.data.point_data.normals = surf['nn']
    mesh.data.cell_data.normals = None
    surf = mlab.pipeline.surface(mesh, figure=fig, reset_zoom=True,
                                 opacity=0.33, color=color)
    surf.actor.property.backface_culling = True
mlab.points3d(hsp[:, 0], hsp[:, 1], hsp[:, 2], color=t_color,
              scale_factor=0.005, opacity=0.25)
mlab.view(45, 90)
