# -*- coding: utf-8 -*-
"""
.. _tut-eeg-mri-coords:

EEG source localization given electrode locations on an MRI
===========================================================

This tutorial explains how to compute the forward operator from EEG data
when the electrodes are in MRI voxel coordinates.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD Style.

import os.path as op

import nibabel
from nilearn.plotting import plot_glass_brain
import numpy as np

import mne
from mne.channels import make_dig_montage, compute_native_head_t
from mne.io.constants import FIFF
from mne.transforms import apply_trans
from mne.viz import plot_alignment

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_T1 = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')

##############################################################################
# Faking an appropriate EEG-MRI dataset
# -------------------------------------
# Let's first manipulate ``sample`` to give us:
#
# - ``eeg_vox``: EEG electrodes in RAS voxel coordinates
# - ``fid_vox``: fiducial (LPA, nasion, RPA) locations in RAS voxel coordinates
# - ``raw``: a Raw instance with only EEG channels.
#
# This is just so we have data to work with -- hopefully with real EEG+MRI
# datasets, these values are obtainable.

trans = mne.read_trans(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif'))
raw = mne.io.read_raw_fif(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'))
# Get fiducials and EEG locations in MRI voxel coords
trans_mm = trans['trans'].copy()
trans_mm[:3, 3] *= 1e3
img = nibabel.load(fname_T1)
vox2mri = img.header.get_vox2ras_tkr()
head_to_vox = np.dot(np.linalg.inv(vox2mri), trans_mm)
fid_vox = apply_trans(head_to_vox, 1000 * np.array([d['r'] for d in sorted(
    [d for d in raw.info['dig'] if d['kind'] == FIFF.FIFFV_POINT_CARDINAL],
    key=lambda d: d['ident'])]))
eeg_vox = apply_trans(head_to_vox, 1000 * np.array(
    [d['r'] for d in raw.info['dig'] if d['kind'] == FIFF.FIFFV_POINT_EEG and
     d['ident'] > 0]))
img, affine = img.get_data().astype(float), img.affine
# Mark them in the MRI with 5x5x5mm cubes
for coord in np.round(np.concatenate([fid_vox, eeg_vox])).astype(int):
    img[tuple(slice(c - 2, c + 2) for c in coord)] = 250
# change "raw" to be more like what you'd acquire: no dig, no channel locations
raw.load_data().pick_types(meg=False, eeg=True, exclude=(), stim=True)
raw.info['dig'] = []
for ch in raw.info['chs']:
    ch['loc'].fill(0.)
del head_to_vox, trans

##############################################################################
# Visualizing the MRI
# -------------------
# Let's take our MRI-with-eeg-locations and adjust the
# affine to MNI space, and plot using nilearn's plot_glass_brain,
# which does a maximum intensity projection (easy to see the fake electrodes).

ras_mni_t = mne.transforms.read_ras_mni_t('sample', subjects_dir)
img_mni = nibabel.Nifti1Image(img, np.dot(ras_mni_t['trans'], affine))
plot_glass_brain(img_mni, cmap='hot_black_bone', threshold=0., black_bg=True,
                 resampling_interpolation='nearest', colorbar=True)
del img, affine, ras_mni_t, img_mni

##########################################################################
# Getting our MRI voxel EEG locations to head (and MRI surface RAS) coords
# ------------------------------------------------------------------------
# For this we will assume that, in addition to ``eeg_vox``, ``fid_vox``, and
# ``raw``, you also have:
#
# - reconstructed your subject using FreeSurfer
# - created an appropriate boundary element model, and
# - created an appropriate source space

bem_dir = op.join(subjects_dir, 'sample', 'bem')
fname_bem = op.join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')
fname_src = op.join(bem_dir, 'sample-oct-6-src.fif')

##############################################################################
# Let's get our MRI channel locations first into RAS space, in meters.
# To do this, we need to use nibabel's header reading functions.
# In MNE the "mri" coordinate frame is the FreeSurfer surface RAS coordinate
# frame, and nibabel calls this coordinate frame "ras_tkr":

img = nibabel.load(fname_T1)
vox2mri = img.header.get_vox2ras_tkr()  # MRI voxel -> FreeSurfer RAS (mm)
vox2mri[:3] /= 1000.  # mm -> m
fid_mri = apply_trans(vox2mri, fid_vox)
eeg_mri = apply_trans(vox2mri, eeg_vox)

##############################################################################
# Now that things are in MRI coordinates, let's make our digitization object
# that will contain the EEG locations, fiducial locations, and coordinate
# frame:

eeg_names = [raw.ch_names[pick] for pick in mne.pick_types(
    raw.info, meg=False, eeg=True, exclude=())]
ch_pos = {ch_name: pos for ch_name, pos in zip(eeg_names, eeg_mri)}
lpa, nasion, rpa = fid_mri
dig_montage = make_dig_montage(ch_pos, nasion, lpa, rpa, coord_frame='mri')
dig_montage.plot()

##############################################################################
# We can then get our transformation from the MRI coordinate frame (where our
# points are defined) to the head coordinate frame from the object:

trans = compute_native_head_t(dig_montage)
print(trans)  # should be mri->head, as the "native" space here is MRI

##############################################################################
# Finally, let's apply this digitization to our dataset:

raw.set_montage(dig_montage)
raw.plot_sensors(show_names=True)

##############################################################################
# Now we can do things like make joint plots of evoked data:

raw.set_eeg_reference(projection=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1)
cov = mne.compute_covariance(epochs, tmax=0.)
evoked = epochs.average()
evoked.plot_joint()

##############################################################################
# Getting a source estimate
# -------------------------
# New we have all of the components we need to compute a forward solution,
# but first we should sanity check things:

plot_alignment(evoked.info, trans=trans, subject='sample',
               subjects_dir=subjects_dir, show_axes=True)

##############################################################################
# Now we can actually compute the forward:

fwd = mne.make_forward_solution(
    evoked.info, trans=trans, src=fname_src, bem=fname_bem, verbose=True)

##############################################################################
# Finally let's compute the inverse and apply it:

inv = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, cov, verbose=True)
stc = mne.minimum_norm.apply_inverse(evoked, inv)
stc.plot(subjects_dir=subjects_dir, initial_time=0.1)
