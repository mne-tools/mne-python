# -*- coding: utf-8 -*-
"""Test ieeg utility functions"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import numpy as np
import pytest

from mne.coreg import get_mni_fiducials
from mne.channels import make_dig_montage
from mne.datasets import testing
from mne.preprocessing.ieeg import make_montage_volume
from mne.transforms import apply_trans
from mne.utils import requires_nibabel, requires_dipy

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"


@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_montage_volume():
    """Test making a montage image based on intracranial electrode positions."""
    import nibabel as nib
    subject_brain = nib.load(subjects_dir / "sample" / "mri" / "brain.mgz")
    # make an info object with three channels with positions
    ch_coords = np.array([[-8.7040273, 17.99938754, 10.29604017],
                          [-14.03007764, 19.69978401, 12.07236939],
                          [-21.1130506, 21.98310911, 13.25658887]])
    ch_pos = dict(zip(['1', '2', '3'], ch_coords / 1000))  # mm -> m
    lpa, nasion, rpa = get_mni_fiducials('sample', subjects_dir)
    montage = make_dig_montage(ch_pos, lpa=lpa['r'], nasion=nasion['r'],
                               rpa=rpa['r'], coord_frame='mri')
    # make fake image based on the info
    CT_data = np.zeros(subject_brain.shape)
    # convert to voxels
    ch_coords_vox = apply_trans(
        np.linalg.inv(subject_brain.header.get_vox2ras_tkr()), ch_coords)
    for (x, y, z) in ch_coords_vox.round().astype(int):
        # make electrode contact hyperintensities
        # first, make the surrounding voxels high intensity
        CT_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = 500
        # then, make the center even higher intensity
        CT_data[x, y, z] = 1000
    CT = nib.Nifti1Image(CT_data, subject_brain.affine)

    elec_image = make_montage_volume(montage, CT, thresh=0.25)
    elec_image_data = np.array(elec_image.dataobj)

    # check elec image, center should be no more than half a voxel away
    for i in range(len(montage.ch_names)):
        assert np.linalg.norm(
            np.array(np.where(elec_image_data == i + 1)
                     ).mean(axis=1) - ch_coords_vox[i]) < 0.5

    # test inputs
    with pytest.raises(ValueError, match='`thresh` must be between 0 and 1'):
        make_montage_volume(montage, CT, thresh=11.)

    bad_montage = montage.copy()
    for d in bad_montage.dig:
        d['coord_frame'] = 99
    with pytest.raises(RuntimeError, match='Coordinate frame not supported'):
        make_montage_volume(bad_montage, CT)
