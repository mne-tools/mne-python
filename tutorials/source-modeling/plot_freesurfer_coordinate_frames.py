"""
================================
Freesurfer and coordinate frames
================================

MNE is tightly integrated with Freesurfer for use of MRI data. This tutorial
is designed to help understand how MRI coordinate frames (and coordinate frames
in general) are handled in MNE.

Starting from MRI
-----------------

Let's start out by looking at the ``sample`` subject MRI. Following standard
Freesurfer convention, we look at ``T1.mgz``, which gets created from the
original MRI ``sample/mri/orig/001.mgz`` when you run
`recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_.
Here we use :meth:`~nibabel.spatialimages.SpatialImage.orthoview` to view it.
"""

import os.path as op
from pprint import pprint

import numpy as np
import nibabel  # standard lib for handling MRI data
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import mne
from mne.transforms import apply_trans
from mne.io.constants import FIFF

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
t1_fname = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')
t1 = nibabel.load(t1_fname)
t1.orthoview()  # nibabel's plotting function

###############################################################################
# Some immediate things are worth noting here. First, is that we will talk
# about the MRI in terms of standard RAS (right-anterior-superior)
# coordinates. If you are unfamiliar with this, see the excellent nibabel
# tutorial :doc:`nibabel:coordinate_systems`. You can see in the plot above
# that the axes are clearly labeled in terms of these coordinates.
#
# Nibabel is taking care of some coordinate frame transformations under the
# hood, but let's see if we can also take these into account. First let's get
# our data as a 3D array and note that it's already a standard size:

data = np.asarray(t1.dataobj)
print(data.shape)  # 256, 256, 256

###############################################################################
# These data are voxels, and relate to real-world X/Y/Z (in RAS orientation)
# location according to the image's affine transformation (noting here that
# nibabel processing everything in units of millimeters):

print(t1.affine)

###############################################################################
# We can then take an arbitrary slice of data, and know where it is in X/Y/Z.
# Here we'll choose some voxel indices along the first axis of our data
# array and visualize it as an image. We'll also figure out, for the center
# pixel of this image (127, 127) what X/Y/Z in millimeters it actually
# corresponds to using the affine:
#
# .. note:: mne.transforms.apply_trans under the hood just effectively does a
#           :func:`~numpy.dot` operation, but takes care of dealing with the
#           fact that the transformation is an affine of shape ``(4, 4)``.

vox = np.array([122, 119, 102])
xyz_ras = apply_trans(t1.affine, vox)


def imshow_mri(data, vox, xyz, suptitle):
    """Show an MRI slice with a voxel annotated."""
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(data[i], vmin=10, vmax=120, cmap='gray', origin='lower')
    ax.axvline(k, color='g')
    ax.axhline(j, color='g')
    for kind, xyz_ in xyz.items():
        text = ax.text(
            k, j, '%s: %d, %d, %d mm ⬊' % ((kind,) + tuple(xyz_)),
            va='baseline', ha='right', color='w')
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()])
    # reorient for our data orientation and label
    ax.set(xlim=[0, 255], ylim=[255, 0],
           xlabel='k', ylabel='j', title='i=%d' % (i,))
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)


imshow_mri(data, vox, dict(RAS=xyz_ras), 'MRI slice')

###############################################################################
# And if you have a point in X/Y/Z (RAS) space that you want the corresponding
# voxel for, you can always get it using the inverse of the affine:

i_, j_, k_ = np.round(apply_trans(
    np.linalg.inv(t1.affine), np.array([0, -16, -18]))).astype(int)
print(i_, j_, k_)  # rounded values, so just close to our originals

###############################################################################
# MRI coordinates in MNE: really Freesurfer surface RAS
# -----------------------------------------------------
# An important difference now arises because MNE makes extensive use of
# *MRI surface RAS* coordinates, which are the coordinates used by Freesurfer
# and therefore also MNE for storing the coordinates of all of its surfaces
# (e.g., white matter, inner/outer skull meshes). This transformation
# is known in the `Freesurfer coordinate frame docs
# <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_ as ``Torig``,
# and in nibabel as :meth:`vox2ras_tkr
# <nibabel.freesurfer.mghformat.MGHHeader.get_vox2ras_tkr>`. We can do the
# same computations as before to figure out where the point is in MRI
# coordinates:

Torig = t1.header.get_vox2ras_tkr()
print(t1.affine)
print(Torig)
xyz_mri = apply_trans(Torig, vox)
imshow_mri(data, vox, dict(MRI=xyz_mri), 'MRI slice')

###############################################################################
# Knowing these relationships and being mindful about transformations, we
# can get from a point in any given space to any other space. Let's start out
# by plotting the Nasion on the MRI:

nasion_mri = mne.coreg.get_mni_fiducials('sample', subjects_dir=subjects_dir)
nasion_mri = [d for d in nasion_mri
              if d['ident'] == FIFF.FIFFV_POINT_NASION][0]
pprint(nasion_mri)  # note it's in MRI coords
nasion_mri = nasion_mri['r'] * 1000  # to mm to please our plotting function
nasion_vox = np.round(
    apply_trans(np.linalg.inv(Torig), nasion_mri)).astype(int)
imshow_mri(data, nasion_vox, dict(MRI=nasion_mri),
           'Nasion estimated from MNI transform')

###############################################################################
# We can also take the digitization point from the MEG data, which is in head
# coordinates, and put it in MRI coordinates:

info = mne.io.read_info(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'))
trans = mne.read_trans(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'))
nasion_head = [d for d in info['dig']
               if d['kind'] == FIFF.FIFFV_POINT_CARDINAL and
               d['ident'] == FIFF.FIFFV_POINT_NASION][0]
pprint(nasion_head)  # head coord frame
# first we transform to MRI, then go to mm
nasion_dig_mri = apply_trans(trans, nasion_head['r']) * 1000
nasion_dig_vox = np.round(
    apply_trans(np.linalg.inv(Torig), nasion_dig_mri)).astype(int)
imshow_mri(data, nasion_dig_vox, dict(MRI=nasion_dig_mri),
           'Nasion transformed from digitization')
