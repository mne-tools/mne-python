"""Test the ieeg projection functions."""
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
from mne.preprocessing.ieeg._projection import _project_sensors_onto_inflated
from mne.datasets import testing
from mne.transforms import _get_trans

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
fname_trans = data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"


@testing.requires_testing_data
def test_project_sensors_onto_brain(tmp_path):
    """Test projecting sensors onto the brain surface."""
    pytest.importorskip("nibabel")
    raw = mne.io.read_raw_fif(fname_raw)
    trans = _get_trans(fname_trans)[0]
    # test informative error for no surface first
    with pytest.raises(RuntimeError, match="requires generating a BEM"):
        project_sensors_onto_brain(raw.info, trans, "sample", subjects_dir=tmp_path)
    brain_surf_fname = tmp_path / "sample" / "bem" / "brain.surf"
    if not brain_surf_fname.parent.is_dir():
        os.makedirs(brain_surf_fname.parent)
    if not brain_surf_fname.is_file():
        copyfile(
            subjects_dir / "sample" / "bem" / "inner_skull.surf",
            brain_surf_fname,
        )
    # now make realistic ECoG grid
    raw.pick_types(meg=False, eeg=True)
    raw.load_data()
    raw.set_eeg_reference([])
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})
    pos = np.zeros((49, 3))
    pos[:, :2] = (
        np.array(np.meshgrid(np.linspace(0, 0.02, 7), np.linspace(0, 0.02, 7)))
        .reshape(2, -1)
        .T
    )
    pos[:, 2] = 0.12
    raw.drop_channels(raw.ch_names[49:])
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos=dict(zip(raw.ch_names[:49], pos)), coord_frame="head"
        )
    )
    raw.info = project_sensors_onto_brain(
        raw.info, trans, "sample", subjects_dir=tmp_path
    )
    # plot to check, should be projected down onto inner skull
    # brain = mne.viz.Brain('sample', subjects_dir=subjects_dir, alpha=0.5,
    #                       surf='white')
    # brain.add_sensors(raw.info, trans=trans)
    test_locs = [
        [0.00149, -0.001588, 0.133029],
        [0.004302, 0.001959, 0.133922],
        [0.008602, 0.00116, 0.133723],
    ]
    montage = raw.get_montage()
    assert montage is not None
    ch_pos = montage.get_positions()["ch_pos"]
    for ch, test_loc in zip(raw.ch_names[:3], test_locs):
        assert_allclose(ch_pos[ch], test_loc, atol=0.01)


@testing.requires_testing_data
def test_project_sensors_onto_inflated(tmp_path):
    """Test projecting sEEG sensors onto an inflated brain surface."""
    pytest.importorskip("nibabel")
    raw = mne.io.read_raw_fif(fname_raw)
    trans = _get_trans(fname_trans)[0]
    for subject in ("sample", "fsaverage"):
        os.makedirs(tmp_path / subject / "surf", exist_ok=True)
        for hemi in ("lh", "rh"):
            # fake white surface for pial
            copyfile(
                subjects_dir / subject / "surf" / f"{hemi}.white",
                tmp_path / subject / "surf" / f"{hemi}.pial",
            )
            copyfile(
                subjects_dir / subject / "surf" / f"{hemi}.curv",
                tmp_path / subject / "surf" / f"{hemi}.curv",
            )
            copyfile(
                subjects_dir / subject / "surf" / f"{hemi}.inflated",
                tmp_path / subject / "surf" / f"{hemi}.inflated",
            )
            if subject == "fsaverage":
                copyfile(
                    subjects_dir / subject / "surf" / f"{hemi}.cortex.patch.flat",
                    tmp_path / subject / "surf" / f"{hemi}.cortex.patch.flat",
                )
                copyfile(
                    subjects_dir / subject / "surf" / f"{hemi}.sphere",
                    tmp_path / subject / "surf" / f"{hemi}.sphere",
                )
    # now make realistic sEEG locations, picked from T1
    raw.pick_types(meg=False, eeg=True)
    raw.load_data()
    raw.set_eeg_reference([])
    raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
    pos = (
        np.array(
            [
                [25.85, 9.04, -5.38],
                [33.56, 9.04, -5.63],
                [40.44, 9.04, -5.06],
                [46.75, 9.04, -6.78],
                [-30.08, 9.04, 28.23],
                [-32.95, 9.04, 37.99],
                [-36.39, 9.04, 46.03],
            ]
        )
        / 1000
    )
    raw.drop_channels(raw.ch_names[len(pos) :])
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos=dict(zip(raw.ch_names, pos)), coord_frame="head"
        )
    )
    proj_info = _project_sensors_onto_inflated(
        raw.info, trans, "sample", subjects_dir=tmp_path
    )
    assert_allclose(
        proj_info["chs"][0]["loc"][:3],
        np.array([0.0555809, 0.0034069, -0.04593032]),
        rtol=0.01,
    )
    # check all on inflated surface
    x_dir = np.array([1.0, 0.0, 0.0])
    head_mri_t = mne.transforms.invert_transform(trans)  # need head->mri
    for hemi in ("lh", "rh"):
        coords, faces = mne.surface.read_surface(
            tmp_path / "sample" / "surf" / f"{hemi}.inflated"
        )
        x_ = coords @ x_dir
        coords -= np.max(x_) * x_dir if hemi == "lh" else np.min(x_) * x_dir
        coords /= 1000  # mm -> m
        for ch in proj_info["chs"]:
            loc = ch["loc"][:3]
            if not np.isnan(loc).any() and (loc[0] <= 0) == (hemi == "lh"):
                assert (
                    np.linalg.norm(
                        coords - mne.transforms.apply_trans(head_mri_t, loc), axis=1
                    ).min()
                    < 1e-16
                )

    # test flat map
    montage = raw.get_montage()
    montage.apply_trans(mne.transforms.invert_transform(trans))
    mri_mni_t = mne.read_talxfm("sample", subjects_dir)
    montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)
    montage.apply_trans(
        mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4))
    )
    raw.set_montage(montage)
    trans = mne.channels.compute_native_head_t(montage)

    flat_proj_info = _project_sensors_onto_inflated(
        raw.info,
        trans=trans,
        subject="fsaverage",
        subjects_dir=tmp_path,
        flat=True,
    )

    # check all on flat surface
    x_dir = np.array([1.0, 0.0, 0.0])
    head_mri_t = mne.transforms.invert_transform(trans)  # need head->mri
    for hemi in ("lh", "rh"):
        coords, faces, _ = mne.surface._read_patch(
            tmp_path / "fsaverage" / "surf" / f"{hemi}.cortex.patch.flat"
        )
        coords = coords[:, [1, 0, 2]]
        coords[:, 1] *= -1
        x_ = coords @ x_dir
        coords -= np.max(x_) * x_dir if hemi == "lh" else np.min(x_) * x_dir
        coords /= 1000  # mm -> m
        for ch in flat_proj_info["chs"]:
            loc = ch["loc"][:3]
            if not np.isnan(loc).any() and (loc[0] <= 0) == (hemi == "lh"):
                assert (
                    np.linalg.norm(
                        coords - mne.transforms.apply_trans(head_mri_t, loc), axis=1
                    ).min()
                    < 1e-16
                )

    # plot to check
    # brain = mne.viz.Brain('fsaverage', subjects_dir=tempdir, alpha=0.5,
    #                       surf='flat')
    # brain.add_sensors(flat_proj_info, trans=trans)
