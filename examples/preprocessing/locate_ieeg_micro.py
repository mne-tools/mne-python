# -*- coding: utf-8 -*-
"""
.. _ex-ieeg-micro:

====================================================
Locating micro-scale intracranial electrode contacts
====================================================

When intracranial electrode contacts are very small, sometimes
the computed tomography (CT) scan is higher resolution than the
magnetic resonance (MR) image and so you want to find the contacts
on the CT without downsampling to the MR resolution. This example
shows how to do this.
"""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import numpy as np
import nibabel as nib
import mne

# path to sample sEEG
misc_path = mne.datasets.misc.data_path()
subjects_dir = misc_path / 'seeg'

# GUI requires pyvista backend
mne.viz.set_3d_backend('pyvistaqt')

# we need three things:
# 1) The electrophysiology file which contains the channels names
# that we would like to associate with positions in the brain
# 2) The CT where the electrode contacts show up with high intensity
# 3) The MR where the brain is best visible (low contrast in CT)
raw = mne.io.read_raw(misc_path / 'seeg' / 'sample_seeg_ieeg.fif')
CT_orig = nib.load(misc_path / 'seeg' / 'sample_seeg_CT.mgz')
T1 = nib.load(misc_path / 'seeg' / 'sample_seeg' / 'mri' / 'T1.mgz')

# we'll also need a head-CT surface RAS transform, this can be faked with an
# identiy matrix but we'll find the fiducials on the CT in freeview (be sure
# to find them in surface RAS (TkReg RAS in freeview) and not scanner RAS
# (RAS in freeview)) (also be sure to note left is generally on the right in
# freeview) and reproduce them here:montage = mne.channels.make_dig_montage(
    nasion=[-28.97, -5.88, -76.40], lpa=[-96.35, -16.26, 17.63],
    rpa=[31.28, -52.95, -0.69], coord_frame='mri')
raw.set_montage(montage, on_missing='ignore')  # haven't located yet!
head_ct_t = mne.transforms.invert_transform(
    mne.channels.compute_native_head_t(montage))

# note: coord_frame = 'mri' is a bit of a misnormer, it is a reference to
# the surface RAS coordinate frame, here it is of the CT


# launch the viewer with only the CT (note, we won't be able to use
# the MR in this case to help determine which brain area the contact is
# in), and use the user interface to find the locations of the contacts
gui = mne.gui.locate_ieeg(raw.info, head_ct_t, CT_orig)

# we'll programmatically mark all the contacts on one electrode shaft
for i, pos in enumerate([(-52.66, -40.84, -26.99), (-55.47, -38.03, -27.92),
                         (-57.68, -36.27, -28.85), (-59.89, -33.81, -29.32),
                         (-62.57, -31.35, -30.37), (-65.13, -29.07, -31.30),
                         (-67.57, -26.26, -31.88)]):
    gui.set_RAS(pos)
    gui.mark_channel(f'LENT {i + 1}')

# finally, the coordinates will be in "head" (unless the trans was faked
# as the identity, in which case they will be in surface RAS of the CT already)
# so we need to convert them to scanner RAS of the aligned CT (which is
# identical to scanner RAS of the MRI) and from there to surface RAS
# of the MRI for viewing using freesurfer recon-all surfaces.
# note that since we didn't fake the head->CT surface RAS transform, we
# could apply the head->mri transform directly but that relies of the
# fiducial points being marked exactly the same on the CT as on the MRI--
# the error from this is not precise enough for intracranial electrophysiology,
# better is to rely on the CT-MR image registration precision

montage = raw.get_montage()  # in head
montage.apply_trans(head_ct_t)  # in CT surface RAS
montage.apply_trans(mne.transforms.Transform(  # in CT voxels
    fro='mri', to='mri_voxel',
    trans=np.linalg.inv(CT_orig.header.get_vox2ras_tkr())))
# to CT scanner RAS == MR scanner RAS
montage.apply_trans(mne.transforms.Transform(
    fro='mri_voxel', to='ras', trans=CT_orig.header.get_vox2ras()))
montage.apply_trans(mne.transforms.Transform(  # to MR voxels
    fro='ras', to='mri_voxel', trans=np.linalg.inv(T1.header.get_vox2ras())))
montage.apply_trans(mne.transforms.Transform(  # to MR voxels
    fro='mri_voxel', to='mri', trans=T1.header.get_vox2ras_tkr()))
head_mri_t = mne.channels.compute_native_head_t(montage)
raw.set_montage(montage)  # converts to head coordinates

brain = mne.viz.Brain(subject='sample_seeg', subjects_dir=subjects_dir)
brain.add_sensors(raw, head_mri_t)
