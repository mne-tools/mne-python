"""Test ieeg volume functions."""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import numpy as np
import pytest

from mne.coreg import get_mni_fiducials
from mne.channels import make_dig_montage
from mne.datasets import testing
from mne.preprocessing.ieeg import make_montage_volume, warp_montage
from mne.transforms import apply_trans, compute_volume_registration

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"


@pytest.mark.slowtest
@testing.requires_testing_data
def test_warp_montage():
    """Test warping an montage based on intracranial electrode positions."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    subject_brain = nib.load(subjects_dir / "sample" / "mri" / "brain.mgz")
    template_brain = nib.load(subjects_dir / "fsaverage" / "mri" / "brain.mgz")
    zooms = dict(translation=10, rigid=10, sdr=10)
    reg_affine, sdr_morph = compute_volume_registration(
        subject_brain,
        template_brain,
        zooms=zooms,
        niter=[3, 3, 3],
        pipeline=("translation", "rigid", "sdr"),
    )
    # make an info object with three channels with positions
    ch_coords = np.array(
        [
            [-8.7040273, 17.99938754, 10.29604017],
            [-14.03007764, 19.69978401, 12.07236939],
            [-21.1130506, 21.98310911, 13.25658887],
        ]
    )
    ch_pos = dict(zip(["1", "2", "3"], ch_coords / 1000))  # mm -> m
    lpa, nasion, rpa = get_mni_fiducials("sample", subjects_dir)
    montage = make_dig_montage(
        ch_pos, lpa=lpa["r"], nasion=nasion["r"], rpa=rpa["r"], coord_frame="mri"
    )
    montage_warped = warp_montage(
        montage, subject_brain, template_brain, reg_affine, sdr_morph
    )
    # checked with nilearn plot from `tut-ieeg-localize`
    # check montage in surface RAS
    ground_truth_warped = np.array(
        [
            [-0.009, -0.00133333, -0.033],
            [-0.01445455, 0.00127273, -0.03163636],
            [-0.022, 0.00285714, -0.031],
        ]
    )
    for i, d in enumerate(montage_warped.dig):
        assert (
            np.linalg.norm(d["r"] - ground_truth_warped[i])  # off by less than 1 cm
            < 0.01
        )

    bad_montage = montage.copy()
    for d in bad_montage.dig:
        d["coord_frame"] = 99
    with pytest.raises(RuntimeError, match="Coordinate frame not supported"):
        warp_montage(bad_montage, subject_brain, template_brain, reg_affine, sdr_morph)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_make_montage_volume():
    """Test making a montage image based on intracranial electrodes."""
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    subject_brain = nib.load(subjects_dir / "sample" / "mri" / "brain.mgz")
    # make an info object with three channels with positions
    ch_coords = np.array(
        [
            [-8.7040273, 17.99938754, 10.29604017],
            [-14.03007764, 19.69978401, 12.07236939],
            [-21.1130506, 21.98310911, 13.25658887],
        ]
    )
    ch_pos = dict(zip(["1", "2", "3"], ch_coords / 1000))  # mm -> m
    lpa, nasion, rpa = get_mni_fiducials("sample", subjects_dir)
    montage = make_dig_montage(
        ch_pos, lpa=lpa["r"], nasion=nasion["r"], rpa=rpa["r"], coord_frame="mri"
    )
    # make fake image based on the info
    CT_data = np.zeros(subject_brain.shape)
    # convert to voxels
    ch_coords_vox = apply_trans(
        np.linalg.inv(subject_brain.header.get_vox2ras_tkr()), ch_coords
    )
    for x, y, z in ch_coords_vox.round().astype(int):
        # make electrode contact hyperintensities
        # first, make the surrounding voxels high intensity
        CT_data[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2] = 500
        # then, make the center even higher intensity
        CT_data[x, y, z] = 1000
    CT = nib.Nifti1Image(CT_data, subject_brain.affine)

    elec_image = make_montage_volume(montage, CT, thresh=0.25)
    elec_image_data = np.array(elec_image.dataobj)

    # check elec image, center should be no more than half a voxel away
    for i in range(len(montage.ch_names)):
        assert (
            np.linalg.norm(
                np.array(np.where(elec_image_data == i + 1)).mean(axis=1)
                - ch_coords_vox[i]
            )
            < 0.5
        )

    # test inputs
    with pytest.raises(ValueError, match="`thresh` must be between 0 and 1"):
        make_montage_volume(montage, CT, thresh=11.0)

    bad_montage = montage.copy()
    for d in bad_montage.dig:
        d["coord_frame"] = 99
    with pytest.raises(RuntimeError, match="Coordinate frame not supported"):
        make_montage_volume(bad_montage, CT)
