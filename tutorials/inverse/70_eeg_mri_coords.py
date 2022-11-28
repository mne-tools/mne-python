# -*- coding: utf-8 -*-
"""
.. _tut-eeg-mri-coords:

===========================================================
EEG source localization given electrode locations on an MRI
===========================================================

This tutorial explains how to compute the forward operator from EEG data when
the electrodes are in MRI voxel coordinates.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import nibabel
from nilearn.plotting import plot_glass_brain
import numpy as np

import mne
from mne.channels import compute_native_head_t, read_custom_montage
from mne.viz import plot_alignment

##############################################################################
# Prerequisites
# -------------
# For this we will assume that you have:
#
# - raw EEG data
# - your subject's MRI reconstrcted using FreeSurfer
# - an appropriate boundary element model (BEM)
# - an appropriate source space (src)
# - your EEG electrodes in Freesurfer surface RAS coordinates, stored
#   in one of the formats :func:`mne.channels.read_custom_montage` supports
#
# Let's set the paths to these files for the ``sample`` dataset, including
# a modified ``sample`` MRI showing the electrode locations plus a ``.elc``
# file corresponding to the points in MRI coords (these were `synthesized
# <https://gist.github.com/larsoner/0ac6fad57e31cb2d9caa77350a9ff366>`__,
# and thus are stored as part of the ``misc`` dataset).

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / 'subjects'
fname_raw = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
bem_dir = subjects_dir / 'sample' / 'bem'
fname_bem = bem_dir / 'sample-5120-5120-5120-bem-sol.fif'
fname_src = bem_dir / 'sample-oct-6-src.fif'

misc_path = mne.datasets.misc.data_path()
fname_T1_electrodes = misc_path / 'sample_eeg_mri' / 'T1_electrodes.mgz'
fname_mon = misc_path / 'sample_eeg_mri' / 'sample_mri_montage.elc'

##############################################################################
# Visualizing the MRI
# -------------------
# Let's take our MRI-with-eeg-locations and adjust the affine to put the data
# in MNI space, and plot using :func:`nilearn.plotting.plot_glass_brain`,
# which does a maximum intensity projection (easy to see the fake electrodes).
# This plotting function requires data to be in MNI space.
# Because ``img.affine`` gives the voxel-to-world (RAS) mapping, if we apply a
# RAS-to-MNI transform to it, it becomes the voxel-to-MNI transformation we
# need. Thus we create a "new" MRI image in MNI coordinates and plot it as:

img = nibabel.load(fname_T1_electrodes)  # original subject MRI w/EEG
ras_mni_t = mne.transforms.read_ras_mni_t('sample', subjects_dir)  # from FS
mni_affine = np.dot(ras_mni_t['trans'], img.affine)  # vox->ras->MNI
img_mni = nibabel.Nifti1Image(img.dataobj, mni_affine)  # now in MNI coords!
plot_glass_brain(img_mni, cmap='hot_black_bone', threshold=0., black_bg=True,
                 resampling_interpolation='nearest', colorbar=True)

##########################################################################
# Getting our MRI voxel EEG locations to head (and MRI surface RAS) coords
# ------------------------------------------------------------------------
# Let's load our :class:`~mne.channels.DigMontage` using
# :func:`mne.channels.read_custom_montage`, making note of the fact that
# we stored our locations in Freesurfer surface RAS (MRI) coordinates.
#
# .. dropdown:: What if my electrodes are in MRI voxels?
#     :color: warning
#     :icon: question
#
#     If you have voxel coordinates in MRI voxels, you can transform these to
#     FreeSurfer surface RAS (called "mri" in MNE) coordinates using the
#     transformations that FreeSurfer computes during reconstruction.
#     ``nibabel`` calls this transformation the ``vox2ras_tkr`` transform
#     and operates in millimeters, so we can load it, convert it to meters,
#     and then apply it::
#
#         >>> pos_vox = ...  # loaded from a file somehow
#         >>> img = nibabel.load(fname_T1)
#         >>> vox2mri_t = img.header.get_vox2ras_tkr()  # voxel -> mri trans
#         >>> pos_mri = mne.transforms.apply_trans(vox2mri_t, pos_vox)
#         >>> pos_mri /= 1000.  # mm -> m
#
#     You can also verify that these are correct (or manually convert voxels
#     to MRI coords) by looking at the points in Freeview or tkmedit.

dig_montage = read_custom_montage(fname_mon, head_size=None, coord_frame='mri')
dig_montage.plot()

##############################################################################
# We can then get our transformation from the MRI coordinate frame (where our
# points are defined) to the head coordinate frame from the object.

trans = compute_native_head_t(dig_montage)
print(trans)  # should be mri->head, as the "native" space here is MRI

##############################################################################
# Let's apply this digitization to our dataset, and in the process
# automatically convert our locations to the head coordinate frame, as
# shown by :meth:`~mne.io.Raw.plot_sensors`.

raw = mne.io.read_raw_fif(fname_raw)
raw.pick_types(meg=False, eeg=True, stim=True, exclude=()).load_data()
raw.set_montage(dig_montage)
raw.plot_sensors(show_names=True)

##############################################################################
# Now we can do standard sensor-space operations like make joint plots of
# evoked data.

raw.set_eeg_reference(projection=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events)
cov = mne.compute_covariance(epochs, tmax=0.)
evoked = epochs['1'].average()  # trigger 1 in auditory/left
evoked.plot_joint()

##############################################################################
# Getting a source estimate
# -------------------------
# New we have all of the components we need to compute a forward solution,
# but first we should sanity check that everything is well aligned:

fig = plot_alignment(
    evoked.info, trans=trans, show_axes=True, surfaces='head-dense',
    subject='sample', subjects_dir=subjects_dir)

##############################################################################
# Now we can actually compute the forward:

fwd = mne.make_forward_solution(
    evoked.info, trans=trans, src=fname_src, bem=fname_bem, verbose=True)

##############################################################################
# Finally let's compute the inverse and apply it:

inv = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, cov, verbose=True)
stc = mne.minimum_norm.apply_inverse(evoked, inv)
brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1)
