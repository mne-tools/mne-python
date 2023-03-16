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
# freeview) and reproduce them here:

# note: coord_frame = 'mri' is a bit of a misnormer, it is a reference to
# the surface RAS coordinate frame, here it is of the CT
montage = mne.channels.make_dig_montage(
    nasion=[-28.97, -5.88, -76.40], lpa=[-96.35, -16.26, 17.63],
    rpa=[31.28, -52.95, -0.69], coord_frame='mri')
raw.set_montage(montage, on_missing='ignore')  # haven't located yet!
head_ct_t = mne.channels.compute_native_head_t(montage)

# launch the viewer with only the CT (note, we won't be able to use
# the MR in this case to help determine which brain area the contact is
# in), and use the user interface to find the locations of the contacts
gui = mne.gui.locate_ieeg(raw.info, head_ct_t, CT_orig)

# we'll programmatically mark all the contacts on one electrode shaft

# finally, the coordinates will be in "head" (unless the trans was faked
# as the identity in which case they will be in surface RAS of the CT already)
# so we need to convert them to scanner RAS of the aligned CT (which is
# identical to scanner RAS of the MRI) and from there to surface RAS
# of the MRI for 