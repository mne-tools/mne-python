# -*- coding: utf-8 -*-
"""Test ieeg utility functions"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import os
from shutil import copyfile
import numpy as np
from numpy.testing import assert_allclose
import pytest

import mne
from mne.preprocessing.ieeg import project_sensors_onto_brain
from mne.preprocessing.ieeg._projection import \
    _project_sensors_onto_inflated
from mne.datasets import testing
from mne.transforms import _get_trans

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
fname_trans = data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"


@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@testing.requires_testing_data
def test_warp_montage_volume():
    """Test warping an montage based on intracranial electrode positions."""
    import nibabel as nib
    subject_brain = nib.load(subjects_dir / "sample" / "mri" / "brain.mgz")
    template_brain = nib.load(subjects_dir / "fsaverage" / "mri" / "brain.mgz")
    zooms = dict(translation=10, rigid=10, sdr=10)
    reg_affine, sdr_morph = compute_volume_registration(
        subject_brain, template_brain, zooms=zooms,
        niter=[3, 3, 3],
        pipeline=('translation', 'rigid', 'sdr'))
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
    ch_coords = np.array([[-8.7040273, 17.99938754, 10.29604017],
                          [-14.03007764, 19.69978401, 12.07236939],
                          [-21.1130506, 21.98310911, 13.25658887]])
    ch_pos = dict(zip(['1', '2', '3'], ch_coords / 1000))  # mm -> m
    lpa, nasion, rpa = get_mni_fiducials('sample', subjects_dir)
    montage = make_dig_montage(ch_pos, lpa=lpa['r'], nasion=nasion['r'],
                               rpa=rpa['r'], coord_frame='mri')
    with pytest.warns(FutureWarning, match='deprecated'):
        montage_warped, image_from, image_to = warp_montage_volume(
            montage, CT, reg_affine, sdr_morph, 'sample',
            subjects_dir_from=subjects_dir, thresh=0.99)
    # checked with nilearn plot from `tut-ieeg-localize`
    # check montage in surface RAS
    ground_truth_warped = np.array([[-0.009, -0.00133333, -0.033],
                                    [-0.01445455, 0.00127273, -0.03163636],
                                    [-0.022, 0.00285714, -0.031]])
    for i, d in enumerate(montage_warped.dig):
        assert np.linalg.norm(  # off by less than 1.5 cm
            d['r'] - ground_truth_warped[i]) < 0.015
    # check image_from
    for idx, contact in enumerate(range(1, len(ch_pos) + 1)):
        voxels = np.array(np.where(np.array(image_from.dataobj) == contact)).T
        assert ch_coords_vox.round()[idx] in voxels
        assert ch_coords_vox.round()[idx] + 5 not in voxels
    # check image_to, too many, just check center
    ground_truth_warped_voxels = np.array(
        [[135.5959596, 161.97979798, 123.83838384],
         [143.11111111, 159.71428571, 125.61904762],
         [150.53982301, 158.38053097, 127.31858407]])
    for i in range(len(montage.ch_names)):
        assert np.linalg.norm(
            np.array(np.where(np.array(image_to.dataobj) == i + 1)
                     ).mean(axis=1) - ground_truth_warped_voxels[i]) < 8
    # test inputs
    with pytest.raises(ValueError, match='`thresh` must be between 0 and 1'):
        with pytest.warns(FutureWarning, match='deprecated'):
            warp_montage_volume(
                montage, CT, reg_affine, sdr_morph, 'sample', thresh=11.)
    with pytest.raises(ValueError, match='subject folder is incorrect'):
        with pytest.warns(FutureWarning, match='deprecated'):
            warp_montage_volume(
                montage, CT, reg_affine, sdr_morph, subject_from='foo',
                subjects_dir_from=subjects_dir)
    CT_unaligned = nib.Nifti1Image(CT_data, template_brain.affine)
    with pytest.raises(RuntimeError, match='not aligned to Freesurfer'):
        with pytest.warns(FutureWarning, match='deprecated'):
            warp_montage_volume(montage, CT_unaligned, reg_affine,
                                sdr_morph, 'sample',
                                subjects_dir_from=subjects_dir)
    bad_montage = montage.copy()
    for d in bad_montage.dig:
        d['coord_frame'] = 99
    with pytest.raises(RuntimeError, match='Coordinate frame not supported'):
        with pytest.warns(FutureWarning, match='deprecated'):
            warp_montage_volume(bad_montage, CT, reg_affine,
                                sdr_morph, 'sample',
                                subjects_dir_from=subjects_dir)

    # check channel not warped
    ch_pos_doubled = ch_pos.copy()
    ch_pos_doubled.update(zip(['4', '5', '6'], ch_coords / 1000))
    doubled_montage = make_dig_montage(
        ch_pos_doubled, lpa=lpa['r'], nasion=nasion['r'],
        rpa=rpa['r'], coord_frame='mri')
    with pytest.warns(RuntimeWarning, match='not assigned'):
        with pytest.warns(FutureWarning, match='deprecated'):
            warp_montage_volume(doubled_montage, CT, reg_affine,
                                None, 'sample', subjects_dir_from=subjects_dir)
