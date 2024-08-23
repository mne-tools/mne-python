# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

from mne._fiff.pick import pick_channels, pick_info, pick_types
from mne.datasets import testing
from mne.io import read_info, read_raw_fil
from mne.preprocessing.hfc import compute_proj_hfc

fil_path = testing.data_path(download=False) / "FIL"
fname_root = "sub-noise_ses-001_task-noise220622_run-001"

io_dir = Path(__file__).parents[2] / "io"
ctf_fname = io_dir / "tests" / "data" / "test_ctf_raw.fif"
fif_fname = io_dir / "tests" / "data" / "test_raw.fif"

# The below channels in the test data do not have positions, set to bad
bads = ["G2-DS-Y", "G2-DS-Z", "G2-DT-Y", "G2-DT-Z", "G2-MW-Y", "G2-MW-Z"]

# TODO: Ignore this warning in all these tests until we deal with this properly
pytestmark = pytest.mark.filterwarnings(
    "ignore:No fiducials.*problems later!:RuntimeWarning",
)


def _unpack_mat(matin):
    """Extract relevant entries from unstructred readmat."""
    data = matin["data"]
    grad = data[0][0]["grad"]
    label = list()
    coil_label = list()
    for ii in range(len(data[0][0]["label"])):
        label.append(str(data[0][0]["label"][ii][0][0]))
    for ii in range(len(grad[0][0]["label"])):
        coil_label.append(str(grad[0][0]["label"][ii][0][0]))

    matout = {
        "label": label,
        "trial": data["trial"][0][0][0][0],
        "coil_label": coil_label,
        "coil_pos": grad[0][0]["coilpos"],
        "coil_ori": grad[0][0]["coilori"],
    }
    return matout


def _angle_between_each(A):
    """Measure the angle between each row vector in a matrix."""
    assert A.ndim == 2
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    d = (A @ A.T).ravel()
    np.clip(d, -1, 1, out=d)
    ang = np.abs(np.arccos(d))
    return ang


@testing.requires_testing_data
@pytest.mark.parametrize("order", [1, 2, 3])
def test_correction(order):
    """Apply HFC and compare to previous computed solutions."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname)
    raw.load_data()
    raw.info["bads"].extend([b for b in bads])
    projs = compute_proj_hfc(raw.info, order=order, accuracy="point")
    raw.add_proj(projs).apply_proj()

    mat = _unpack_mat(loadmat(fil_path / f"{fname_root}_hfc_l{order}.mat"))

    proj_list = projs[0]["data"]["col_names"]
    picks = pick_channels(raw.ch_names, proj_list, ordered=True)
    mat_list = mat["coil_label"]
    mat_inds = pick_channels(mat_list, proj_list, ordered=True)

    want = mat["trial"][mat_inds]
    got = raw.copy().add_proj(projs).apply_proj()[picks, 0:300][0] * 1e15
    assert_allclose(got, want, rtol=1e-7)

    # Now with default accuracy: not super close with tol but corr is good
    projs = compute_proj_hfc(raw.info, order=order)
    got = raw.copy().add_proj(projs).apply_proj()[picks, 0:300][0] * 1e15
    corr = np.corrcoef(got.ravel(), want.ravel())[0, 1]
    assert 0.999999 < corr <= 1.0


@testing.requires_testing_data
def test_l1_basis_orientations():
    """Test that angles between the basis components matches orientations."""
    binname = fil_path / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
    raw = read_raw_fil(binname)
    raw.info["bads"].extend([b for b in bads])
    projs = compute_proj_hfc(raw.info, accuracy="point")
    basis = np.hstack([p["data"]["data"].T for p in projs])
    picks = pick_types(raw.info, meg="mag")
    assert len(picks) == 68
    assert basis.shape == (len(picks), 3)
    ang_model = _angle_between_each(basis)
    n_ang = len(picks) ** 2
    assert ang_model.shape == (n_ang,)

    chs = pick_info(raw.info, picks)["chs"]
    ori_sens = np.array([ch["loc"][-3:] for ch in chs])
    # match the normalization that our projectors get
    ori_sens /= np.linalg.norm(ori_sens, axis=0, keepdims=True)
    assert ori_sens.shape == (len(picks), 3)
    ang_sens = _angle_between_each(ori_sens)
    assert ang_sens.shape == (n_ang,)

    assert_allclose(ang_sens, ang_model, atol=1e-7)


def test_ref_degenerate():
    """Test reference channel handling and degenerate conditions."""
    info = read_info(ctf_fname)
    # exclude ref by default
    projs = compute_proj_hfc(info)
    meg_names = [
        info["ch_names"][pick]
        for pick in pick_types(info, meg=True, ref_meg=False, exclude=[])
    ]
    assert len(projs) == 3
    assert projs[0]["desc"] == "HFC: l=1 m=-1"
    assert projs[1]["desc"] == "HFC: l=1 m=0"
    assert projs[2]["desc"] == "HFC: l=1 m=1"
    assert projs[0]["data"]["col_names"] == meg_names
    meg_ref_names = [
        info["ch_names"][pick]
        for pick in pick_types(info, meg=True, ref_meg=True, exclude=[])
    ]
    projs = compute_proj_hfc(info, picks=("meg", "ref_meg"))
    assert projs[0]["data"]["col_names"] == meg_ref_names

    # Degenerate
    info = read_info(fif_fname)
    compute_proj_hfc(info)  # smoke test
    with pytest.raises(ValueError, match="Only.*could be interpreted as MEG"):
        compute_proj_hfc(info, picks=[0, 330])  # one MEG, one EEG
    info["chs"][0]["loc"][:] = np.nan  # first MEG proj
    with pytest.raises(ValueError, match="non-finite projectors"):
        compute_proj_hfc(info)
    info_eeg = pick_info(info, pick_types(info, meg=False, eeg=True))
    with pytest.raises(ValueError, match=r"picks \(\'meg\'\) could not be"):
        compute_proj_hfc(info_eeg)
