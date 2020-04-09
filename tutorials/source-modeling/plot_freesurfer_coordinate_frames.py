"""
================================
Freesurfer and coordinate frames
================================

This tutorial explains how MRI coordinate frames (and coordinate frames in
general) are handled in MNE-Python, and how MNE-Python integrates with
Freesurfer for handling MRI data.

As usual we'll start by importing the necessary packages; for this tutorial
that includes :mod:`nibabel` to handle loading the MRI images (MNE-Python also
uses :mod:`nibabel` under the hood). We'll also use a special :mod:`Matplotlib
<matplotlib.patheffects>` function for adding outlines to text, so that text is
readable on top of an MRI image.
"""

import os

import numpy as np
import nibabel
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import mne
from mne.transforms import apply_trans
from mne.io.constants import FIFF

###############################################################################
# Starting from MRI
# -----------------
#
# Let's start out by looking at the ``sample`` subject MRI. Following standard
# Freesurfer convention, we look at :file:`T1.mgz`, which gets created from the
# original MRI :file:`sample/mri/orig/001.mgz` when you run the Freesurfer
# command `recon-all <https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all>`_.
# Here we use :mod:`nibabel` to load the T1 image, and the resulting object's
# :meth:`~nibabel.spatialimages.SpatialImage.orthoview` method to view it.

data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')
t1_fname = os.path.join(subjects_dir, 'sample', 'mri', 'T1.mgz')
t1 = nibabel.load(t1_fname)
t1.orthoview()

###############################################################################
# Notice that the axes in the
# :meth:`~nibabel.spatialimages.SpatialImage.orthoview` figure are labeled
# L-R, S-I, and P-A. These reflect the standard RAS (right-anterior-superior)
# coordinate system that is widely used in MRI imaging. If you are unfamiliar
# with RAS coordinates, see the excellent nibabel tutorial
# :doc:`nibabel:coordinate_systems`.
#
# Nibabel already takes care of some coordinate frame transformations under the
# hood, so let's do it manually so we understand what is happening. First let's
# get our data as a 3D array and note that it's already a standard size:

data = np.asarray(t1.dataobj)
print(data.shape)

###############################################################################
# These data are voxel intensity values (unsigned integers in the range 0-255).
# Their positions in the voxel array are related to real-world ``(x, y, z)``
# positions (in RAS orientation) by the image's affine transformation (noting
# here that :mod:`nibabel` processes everything in units of millimeters):

print(t1.affine)

###############################################################################
# .. sidebar:: Under the hood
#
#     ``mne.transforms.apply_trans`` effectively does a matrix multiplication
#     (i.e., :func:`numpy.dot`), with a little extra work to handle the shape
#     mismatch (the affine has shape ``(4, 4)`` because it includes a
#     *translation*, which is applied separately).
#
# This allows us to take an arbitrary voxel or slice of data and know where it
# is in real-world ``(x, y, z)``, by applying the affine transformation to the
# voxel coordinates.

vox = np.array([122, 119, 102])
xyz_ras = apply_trans(t1.affine, vox)
print('Our voxel has real-world coordinates {}, {}, {} (mm)'
      .format(*np.round(xyz_ras, 3)))

###############################################################################
# If you have a point ``(x, y, z)`` in RAS space and you want the
# corresponding voxel number, you can get it using the inverse of the
# affine. This involves some rounding, so it's possible to end up off by one
# voxel if you're not careful:

ras_coords_mm = np.array([1, -17, -18])
inv_affine = np.linalg.inv(t1.affine)
i_, j_, k_ = np.round(apply_trans(inv_affine, ras_coords_mm)).astype(int)
print('Our real-world coordinates correspond to voxel ({}, {}, {})'
      .format(i_, j_, k_))

###############################################################################
# Let's write a short function to visualize where our voxel lies in a saggital
# image, and annotate it in RAS space (rounded to the nearest millimeter):


def imshow_mri(data, vox, xyz, suptitle):
    """Show an MRI slice with a voxel annotated."""
    i, j, k = vox
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(data[i], vmin=10, vmax=120, cmap='gray', origin='lower')
    ax.axvline(k, color='y')
    ax.axhline(j, color='y')
    for kind, coords in xyz.items():
        annotation = ('{}: {}, {}, {} mm ⬊'
                      .format(kind, *np.round(coords).astype(int)))
        text = ax.text(k, j, annotation, va='baseline', ha='right',
                       color=(1, 1, 0.7))
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()])
    # reorient for our data orientation and label
    ax.set(xlim=[0, 255], ylim=[255, 0], xlabel='k', ylabel='j',
           title='i={}'.format(i))
    fig.suptitle(suptitle)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)


imshow_mri(data, vox, dict(RAS=xyz_ras), 'MRI slice')

###############################################################################
# Notice that the axis scales (``i``, ``j``, and ``k``) are still in voxels
# (ranging from 0-255); it's only the annotation text that we've translated
# into real-world RAS in millimeters.
#
#
# MRI coordinates in MNE-Python are really "Freesurfer surface RAS"
# -----------------------------------------------------------------
#
# While :mod:`nibabel` uses standard RAS ``(x, y, z)`` coordinates, Freesurfer
# uses *MRI surface RAS* coordinates; the transform from voxels to Freesurfer
# MRI surface RAS coordinate frame is known in the `Freesurfer documentation
# <https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems>`_ as ``Torig``,
# and in nibabel as :meth:`vox2ras_tkr
# <nibabel.freesurfer.mghformat.MGHHeader.get_vox2ras_tkr>`.
# Since MNE-Python uses Freesurfer extensively for surface computations (e.g.,
# white matter, inner/outer skull meshes), internally MNE-Python often uses the
# Freesurfer surface RAS coordinate system (not the :mod:`nibabel` RAS system).
# We can do similar computations as before to figure out where the point is in
# Freesurfer MRI coordinates:

Torig = t1.header.get_vox2ras_tkr()
print(t1.affine)
print(Torig)
xyz_mri = apply_trans(Torig, vox)
imshow_mri(data, vox, dict(MRI=xyz_mri), 'MRI slice')

###############################################################################
# Knowing these relationships and being mindful about transformations, we
# can get from a point in any given space to any other space. Let's start out
# by plotting the Nasion on a saggital MRI slice:

fiducials = mne.coreg.get_mni_fiducials('sample', subjects_dir=subjects_dir)
nasion_mri = [d for d in fiducials if d['ident'] == FIFF.FIFFV_POINT_NASION][0]
print(nasion_mri)  # note it's in MRI coords

###############################################################################
# When we print the nasion, it displays as a ``DigPoint`` and shows its
# coordinates in millimeters, but beware that the underlying data is
# :ref:`actually stored in meters <units>`,
# so before transforming and plotting we'll convert to millimeters:

nasion_mri = nasion_mri['r'] * 1000  # meters → millimeters
nasion_vox = np.round(
    apply_trans(np.linalg.inv(Torig), nasion_mri)).astype(int)
imshow_mri(data, nasion_vox, dict(MRI=nasion_mri),
           'Nasion estimated from MNI transform')

###############################################################################
# We can also take the digitization point from the MEG data, which is in the
# "head" coordinate frame:

info = mne.io.read_info(
    os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'))
nasion_head = [d for d in info['dig'] if
               d['kind'] == FIFF.FIFFV_POINT_CARDINAL and
               d['ident'] == FIFF.FIFFV_POINT_NASION][0]
print(nasion_head)  # note it's in "head" coordinates

###############################################################################
# To convert from head coordinate frame to MRI, we first apply the transform
# from a :file:`trans` file (typically created with the MNE-Python
# coregistration GUI), then convert meters → millimeters, and finally apply the
# inverse of ``Torig`` to get to voxels:

trans = mne.read_trans(
    os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'))

# first we transform from head to MRI, and *then* convert to millimeters
nasion_dig_mri = apply_trans(trans, nasion_head['r']) * 1000

# ...then we can use Torig to convert MRI to RAS:
nasion_dig_vox = np.round(
    apply_trans(np.linalg.inv(Torig), nasion_dig_mri)).astype(int)
imshow_mri(data, nasion_dig_vox, dict(MRI=nasion_dig_mri),
           'Nasion transformed from digitization')
