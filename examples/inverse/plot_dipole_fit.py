# -*- coding: utf-8 -*-
"""
===============
Do a dipole fit
===============

This shows how to fit a dipole using mne-python.

For a comparison of fits between MNE-C and mne-python, see:

    https://gist.github.com/Eric89GXL/ca55f791200fe1dc3dd2

"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op

import mne

print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'sample', 'surf', 'lh.white')

# Let's localize the N100m (using MEG only)
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                          baseline=(None, 0))
evoked = mne.pick_types_evoked(evoked, meg=True, eeg=False)
evoked.crop(0.07, 0.08)

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

###############################################################################
# Show result on 3D source space
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab

rh_points, rh_faces = mne.read_surface(fname_surf_lh)
rh_points /= 1000.
coord_trans = mne.transforms.invert_transform(mne.read_trans(fname_trans))
coord_trans = coord_trans['trans']
rh_points = mne.transforms.apply_trans(coord_trans, rh_points)
mlab.figure(size=(600, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

# show brain surface after proper coordinate system transformation
brain_surface = mne.read_bem_surfaces(fname_bem, patch_stats=True)[0]
points = brain_surface['rr']
faces = brain_surface['tris']
points = mne.transforms.apply_trans(coord_trans, points)
mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                     faces, color=(1, 1, 0), opacity=0.1)

# show one cortical surface
mlab.triangular_mesh(rh_points[:, 0], rh_points[:, 1], rh_points[:, 2],
                     rh_faces, color=(0.7, ) * 3, opacity=0.3)

# show dipole as small cones
dipoles = mlab.quiver3d(dip.pos[:, 0], dip.pos[:, 1], dip.pos[:, 2],
                        dip.ori[:, 0], dip.ori[:, 1], dip.ori[:, 2],
                        opacity=0.8, scale_factor=4e-3, scalars=dip.times,
                        mode='cone', colormap='RdBu')
# revert colormap
dipoles.module_manager.scalar_lut_manager.reverse_lut = True
mlab.colorbar(dipoles, title='Dipole fit time (ms)')

# proper 3D orientation
mlab.get_engine().scenes[0].scene.x_plus_view()
