# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mne
from mne import (
    get_volume_labels_from_aseg,
    head_to_mni,
    read_freesurfer_lut,
    read_talxfm,
    vertex_to_mni,
)
from mne._freesurfer import (
    _check_subject_dir,
    _estimate_talxfm_rigid,
    _get_mgz_header,
    read_lta,
)
from mne.datasets import testing
from mne.transforms import _angle_between_quats, _get_trans, apply_trans, rot_to_quat

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
fname_mri = data_path / "subjects" / "sample" / "mri" / "T1.mgz"
aseg_fname = data_path / "subjects" / "sample" / "mri" / "aseg.mgz"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
rng = np.random.RandomState(0)


@testing.requires_testing_data
def test_check_subject_dir():
    """Test checking for a Freesurfer recon-all subject directory."""
    _check_subject_dir("sample", subjects_dir)
    with pytest.raises(ValueError, match="subject folder is incorrect"):
        _check_subject_dir("foo", data_path)


@testing.requires_testing_data
def test_mgz_header():
    """Test MGZ header reading."""
    nib = pytest.importorskip("nibabel")
    header = _get_mgz_header(fname_mri)
    mri_hdr = nib.load(fname_mri).header
    assert_allclose(mri_hdr.get_data_shape(), header["dims"])
    assert_allclose(mri_hdr.get_vox2ras_tkr(), header["vox2ras_tkr"])
    assert_allclose(mri_hdr.get_ras2vox(), np.linalg.inv(header["vox2ras"]))


@testing.requires_testing_data
def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates."""
    pytest.importorskip("nibabel")
    # obtained using "tksurfer (sample) (l/r)h white"
    vertices = [100960, 7620, 150549, 96761]
    coords = np.array(
        [
            [-60.86, -11.18, -3.19],
            [-36.46, -93.18, -2.36],
            [-38.00, 50.08, -10.61],
            [47.14, 8.01, 46.93],
        ]
    )
    hemis = [0, 0, 0, 1]
    coords_2 = vertex_to_mni(vertices, hemis, "sample", subjects_dir)
    # less than 1mm error
    assert_allclose(coords, coords_2, atol=1.0)


@testing.requires_testing_data
def test_head_to_mni():
    """Test conversion of aseg vertices to MNI coordinates."""
    # obtained using freeview
    coords = (
        np.array(
            [
                [22.52, 11.24, 17.72],
                [22.52, 5.46, 21.58],
                [16.10, 5.46, 22.23],
                [21.24, 8.36, 22.23],
            ]
        )
        / 1000.0
    )

    xfm = read_talxfm("sample", subjects_dir)
    coords_MNI = apply_trans(xfm["trans"], coords) * 1000.0

    mri_head_t, _ = _get_trans(trans_fname, "mri", "head", allow_none=False)

    # obtained from sample_audvis-meg-oct-6-mixed-fwd.fif
    coo_right_amygdala = np.array(
        [
            [0.01745682, 0.02665809, 0.03281873],
            [0.01014125, 0.02496262, 0.04233755],
            [0.01713642, 0.02505193, 0.04258181],
            [0.01720631, 0.03073877, 0.03850075],
        ]
    )
    coords_MNI_2 = head_to_mni(coo_right_amygdala, "sample", mri_head_t, subjects_dir)
    # less than 1mm error
    assert_allclose(coords_MNI, coords_MNI_2, atol=10.0)


@testing.requires_testing_data
def test_vertex_to_mni_fs_nibabel(monkeypatch):
    """Test equivalence of vert_to_mni for nibabel and freesurfer."""
    pytest.importorskip("nibabel")
    n_check = 1000
    subject = "sample"
    vertices = rng.randint(0, 100000, n_check)
    hemis = rng.randint(0, 1, n_check)
    coords = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    read_mri = mne._freesurfer._read_mri_info
    monkeypatch.setattr(
        mne._freesurfer,
        "_read_mri_info",
        lambda *args, **kwargs: read_mri(*args, use_nibabel=True, **kwargs),
    )
    coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    # less than 0.1 mm error
    assert_allclose(coords, coords_2, atol=0.1)


def test_read_lta(tmp_path):
    """Test reading a Freesurfer linear transform array file."""
    with open(tmp_path / "test.lta", "w") as fid:
        fid.write(
            """type      = 0 # LINEAR_VOX_TO_VOX
                     nxforms   = 1
                     mean      = 0.0000 0.0000 0.0000
                     sigma     = 1.0000
                     1 4 4
                     0.99221027 -0.05494503  0.11180324 -3.84350586
                     0.05233596  0.99828744 0.02614108 -9.77523804
                     -0.11304809 -0.02008611 0.99338663 15.25457001
                     0 0 0 1
                     src volume info
                     valid = 1  # volume info valid
                     filename = tmp.mgz
                     volume = 256 256 256
                     voxelsize = 1 1 1
                     xras   = -1 0 0
                     yras   = 0 0 -1
                     zras   = 0 1 0
                     cras   = -1.19374 -3.31686 3.25835
                     dst volume info
                     valid = 1  # volume info valid
                     filename = tmp.mgz
                     volume = 256 256 256
                     voxelsize = 1 1 1
                     xras   = -1 0 0
                     yras   = 0 0 -1
                     zras   = 0 1 0
                     cras   = -1.19374 -3.31686 3.25835"""
        )
    assert_array_equal(
        read_lta(tmp_path / "test.lta"),
        np.array(
            [
                [0.99221027, -0.05494503, 0.11180324, -3.84350586],
                [0.05233596, 0.99828744, 0.02614108, -9.77523804],
                [-0.11304809, -0.02008611, 0.99338663, 15.25457001],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    # test when dst volume != src_volume
    with open(tmp_path / "test2.lta", "w") as fid:
        fid.write(
            """type      = 0 # LINEAR_VOX_TO_VOX
                     nxforms   = 1
                     mean      = 0.0000 0.0000 0.0000
                     sigma     = 1.0000
                     1 4 4
                     0.41397345  -0.02919456  -0.00069703  26.37020874
                     -0.02894894 -0.40985453  -0.06119149 212.38204956
                     0.00361269   0.0611503   -0.41046342 203.33338928
                     0 0 0 1
                     src volume info
                     valid = 1  # volume info valid
                     filename = tmp2.mgz
                     volume = 512 385 512
                     voxelsize = 0.41499999 0.41541821 0.41499999
                     xras   = -1 0 0
                     yras   = 0 0 1
                     zras   = 0 -1 0
                     cras   = -106.23999786 105.82500458 -79.55259705
                     dst volume info
                     valid = 1  # volume info valid
                     filename = tmp.mgz
                     volume = 256 256 256
                     voxelsize = 1 1 1
                     xras   = -1 0 0
                     yras   = 0 0 -1
                     zras   = 0 1 0
                     cras   = -3.68961334 -0.12011719 3.4160614"""
        )
    assert_allclose(
        read_lta(tmp_path / "test2.lta"),
        np.array(
            [
                [0.99752641, -0.07034834, -0.00167959, -236.00043542],
                [0.06968626, 0.98660704, 0.14730093, 189.09766694],
                [-0.00870528, -0.14735012, 0.98906851, 329.7632126],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        atol=1e-8,
    )


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname",
    [
        None,
        Path(mne.__file__).parent / "data" / "FreeSurferColorLUT.txt",
    ],
)
def test_read_freesurfer_lut(fname, tmp_path):
    """Test reading volume label names."""
    pytest.importorskip("nibabel")
    atlas_ids, colors = read_freesurfer_lut(fname)
    assert list(atlas_ids).count("Brain-Stem") == 1
    assert len(colors) == len(atlas_ids) == 1266
    label_names, label_colors = get_volume_labels_from_aseg(
        aseg_fname, return_colors=True
    )
    assert isinstance(label_names, list)
    assert isinstance(label_colors, list)
    assert label_names.count("Brain-Stem") == 1
    for c in label_colors:
        assert isinstance(c, np.ndarray)
        assert c.shape == (4,)
    assert len(label_names) == len(label_colors) == 46
    with pytest.raises(ValueError, match="must be False"):
        get_volume_labels_from_aseg(aseg_fname, return_colors=True, atlas_ids=atlas_ids)
    label_names_2 = get_volume_labels_from_aseg(aseg_fname, atlas_ids=atlas_ids)
    assert label_names == label_names_2
    # long name (only test on one run)
    if fname is not None:
        return
    fname = tmp_path / "long.txt"
    names = [
        "Anterior_Cingulate_and_Medial_Prefrontal_Cortex-" + hemi
        for hemi in ("lh", "rh")
    ]
    ids = np.arange(1, len(names) + 1)
    colors = [(id_,) * 4 for id_ in ids]
    with open(fname, "w") as fid:
        for name, id_, color in zip(names, ids, colors):
            out_color = " ".join(f"{x:03}" for x in color)
            line = f"{id_}    {name} {out_color}\n"
            fid.write(line)
    lut, got_colors = read_freesurfer_lut(fname)
    assert len(lut) == len(got_colors) == len(names) == len(ids)
    for name, id_, color in zip(names, ids, colors):
        assert name in lut
        assert name in got_colors
        assert_array_equal(got_colors[name][:3], color[:3])
        assert lut[name] == id_
    with open(fname, "w") as fid:
        for name, id_, color in zip(names, ids, colors):
            out_color = " ".join(f"{x:03}" for x in color[:3])  # wrong length!
            line = f"{id_}    {name} {out_color}\n"
            fid.write(line)
    with pytest.raises(RuntimeError, match="formatted"):
        read_freesurfer_lut(fname)


@testing.requires_testing_data
def test_talxfm_rigid():
    """Test that talxfm_rigid gives reasonable results."""
    rigid = _estimate_talxfm_rigid("fsaverage", subjects_dir=subjects_dir)
    assert_allclose(rigid, np.eye(4), atol=1e-6)
    rigid = _estimate_talxfm_rigid("sample", subjects_dir=subjects_dir)
    assert_allclose(np.linalg.norm(rigid[:3, :3], axis=1), 1.0, atol=1e-6)
    move = 1000 * np.linalg.norm(rigid[:3, 3])
    assert 30 < move < 70
    ang = np.rad2deg(_angle_between_quats(rot_to_quat(rigid[:3, :3])))
    assert 20 < ang < 25
