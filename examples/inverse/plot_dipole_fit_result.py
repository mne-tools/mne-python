"""
==============================================================
Reading a .dip file form xfit and view with source space in 3D
==============================================================

Here the .dip file was generated with the mne_dipole_fit command.

Detailed unix command is :

$mne_dipole_fit --meas sample_audvis-ave.fif --set 1 --meg --tmin 40 --tmax 95 \
    --bmin -200 --bmax 0 --noise sample_audvis-cov.fif \
    --bem ../../subjects/sample/bem/sample-5120-bem-sol.fif \
    --origin 0:0:40 --mri sample_audvis-meg-oct-6-fwd.fif \
    --dip sample_audvis_set1.dip

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import mne
from mne.datasets import sample

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
dip_fname = data_path + '/MEG/sample/sample_audvis_set1.dip'
bem_fname = data_path + '/subjects/sample/bem/sample-5120-bem-sol.fif'

brain_surface = mne.read_bem_surfaces(bem_fname, add_geom=True)[0]
points = brain_surface['rr']
faces = brain_surface['tris']

fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

# read dipoles
time, pos, amplitude, ori, gof = mne.read_dip(dip_fname)

print "Time (ms): %s" % time
print "Amplitude (nAm): %s" % amplitude
print "GOF (%%): %s" % gof

# only plot those for which GOF is above 50%
pos = pos[gof > 50.]
ori = ori[gof > 50.]
time = time[gof > 50.]

###############################################################################
# Show result on 3D source space
try:
    from enthought.mayavi import mlab
except:
    from mayavi import mlab

lh_points = src[0]['rr']
lh_faces = src[0]['use_tris']
mlab.figure(size=(600, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

# show brain surface after proper coordinate system transformation
points = brain_surface['rr']
faces = brain_surface['tris']
coord_trans = fwd['mri_head_t']['trans']
points = np.dot(coord_trans[:3,:3], points.T).T + coord_trans[:3,-1]
mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
                     faces, color=(1, 1, 0), opacity=0.3)

# show one cortical surface
mlab.triangular_mesh(lh_points[:, 0], lh_points[:, 1], lh_points[:, 2],
                     lh_faces, color=(0.7, ) * 3)

# show dipole as small cones
dipoles = mlab.quiver3d(pos[:,0], pos[:,1], pos[:,2],
                        ori[:,0], ori[:,1], ori[:,2],
                        opacity=1., scale_factor=4e-4, scalars=time,
                        mode='cone')
mlab.colorbar(dipoles, title='Dipole fit time (ms)')

# proper 3D orientation
mlab.get_engine().scenes[0].scene.x_plus_view()
