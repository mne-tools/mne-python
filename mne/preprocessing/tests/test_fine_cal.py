# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from mne import pick_types
from mne._fiff.tag import _loc_to_coil_trans
from mne.datasets import testing
from mne.io import read_raw_ctf, read_raw_fif, read_raw_fil, read_raw_kit
from mne.preprocessing import (
    compute_fine_calibration,
    maxwell_filter,
    read_fine_calibration,
    write_fine_calibration,
)
from mne.preprocessing.tests.test_maxwell import _assert_shielding
from mne.transforms import _angle_dist_between_rigid
from mne.utils import catch_logging, object_diff

# Define fine calibration filepaths
data_path = testing.data_path(download=False)
fine_cal_fname = data_path / "SSS" / "sss_cal_3053.dat"
fine_cal_fname_3d = data_path / "SSS" / "sss_cal_3053_3d.dat"
erm_fname = data_path / "SSS" / "141027_cropped_90Hz_raw.fif"
ctc = data_path / "SSS" / "ct_sparse.fif"
cal_mf_fname = data_path / "SSS" / "141027.dat"
triux_path = data_path / "SSS" / "TRIUX"
tri_fname = triux_path / "triux_bmlhus_erm_raw.fif"
tri_cal_fname = triux_path / "sss_cal_BMLHUS.dat"
ctf_fname_continuous = data_path / "CTF" / "testdata_ctf.ds"
io_dir = Path(__file__).parents[2] / "io"
kit_dir = io_dir / "kit" / "tests" / "data"
sqd_path = kit_dir / "test.sqd"
mrk_path = kit_dir / "test_mrk.sqd"
elp_path = kit_dir / "test_elp.txt"
hsp_path = kit_dir / "test_hsp.txt"
fil_fname = data_path / "FIL" / "sub-noise_ses-001_task-noise220622_run-001_meg.bin"
td_mark = testing._pytest_mark()


@pytest.mark.parametrize("fname", (cal_mf_fname, fine_cal_fname, fine_cal_fname_3d))
@testing.requires_testing_data
def test_fine_cal_io(tmp_path, fname):
    """Test round trip reading/writing of fine calibration .dat file."""
    temp_fname = tmp_path / "fine_cal_temp.dat"
    # Load fine calibration file
    fine_cal_dict = read_fine_calibration(fname)

    # Save temp version of fine calibration file
    write_fine_calibration(temp_fname, fine_cal_dict)
    fine_cal_dict_reload = read_fine_calibration(temp_fname)

    # Load temp version of fine calibration file and compare hashes
    assert object_diff(fine_cal_dict, fine_cal_dict_reload) == ""


@testing.requires_testing_data
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("VectorView", marks=pytest.mark.ultraslowtest),  # ~7s
        pytest.param("TRIUX", marks=pytest.mark.ultraslowtest),  # ~14s
    ],
)
def test_compute_fine_cal(kind):
    """Test computing fine calibration coefficients."""
    cl = dict(mag=(0.99, 1.01), grad=(0.99, 1.01))
    if kind == "VectorView":
        erm = erm_fname
        cal = cal_mf_fname
        err_limit = 5
        angle_limit = 5
        gwoma = [66, 68]
        ggoma = [55, 150]
        ggwma = [60, 86]
        sfs = [26, 27, 61, 63, 61, 63, 68, 70]
        cl3 = [0.6, 0.7]
    else:
        assert kind == "TRIUX"
        erm = tri_fname
        cal = tri_cal_fname
        err_limit = 10
        angle_limit = 10
        cl["grad"] = (0.0, 0.1)
        gwoma = [48, 52]
        ggoma = [13, 67]
        ggwma = [13, 120]
        sfs = [34, 35, 27, 28, 50, 53, 75, 79]  # ours is better!
        cl3 = [-0.3, -0.1]
    raw = read_raw_fif(erm)
    want_cal = read_fine_calibration(cal)
    with pytest.raises(ValueError, match="err_limit.*greater.*0"):
        compute_fine_calibration(raw, err_limit=-1)
    with pytest.raises(ValueError, match="angle_limit.*greater.*0"):
        compute_fine_calibration(raw, angle_limit=-1)
    got_cal, counts = compute_fine_calibration(
        raw,
        cross_talk=ctc,
        n_imbalance=1,
        err_limit=err_limit,
        angle_limit=angle_limit,
        verbose=True,
    )
    assert counts == 1
    assert set(got_cal.keys()) == set(want_cal.keys())
    assert got_cal["ch_names"] == want_cal["ch_names"]
    # in practice these should never be exactly 1.
    assert sum([(ic == 1.0).any() for ic in want_cal["imb_cals"]]) == 0
    assert sum([(ic == 1.0).any() for ic in got_cal["imb_cals"]]) < 2

    got_imb = np.array(got_cal["imb_cals"], float)
    want_imb = np.array(want_cal["imb_cals"], float)
    assert got_imb.shape == want_imb.shape == (306, 1)
    got_imb, want_imb = got_imb[:, 0], want_imb[:, 0]

    meg_picks = pick_types(raw.info, meg=True, ref_meg=False, exclude=())
    orig_locs = np.array([raw.info["chs"][pick]["loc"] for pick in meg_picks])
    want_locs = want_cal["locs"]
    got_locs = got_cal["locs"]
    assert want_locs.shape == got_locs.shape

    orig_trans = _loc_to_coil_trans(orig_locs)
    want_trans = _loc_to_coil_trans(want_locs)
    got_trans = _loc_to_coil_trans(got_locs)
    want_orig_angles, want_orig_dist = _angle_dist_between_rigid(
        want_trans,
        orig_trans,
        angle_units="deg",
        distance_units="mm",
    )
    got_want_angles, got_want_dist = _angle_dist_between_rigid(
        got_trans,
        want_trans,
        angle_units="deg",
        distance_units="mm",
    )
    got_orig_angles, got_orig_dist = _angle_dist_between_rigid(
        got_trans,
        orig_trans,
        angle_units="deg",
        distance_units="mm",
    )
    assert_array_less(got_want_dist, 0.01)
    assert_array_less(got_orig_dist, 0.01)
    for key in ("mag", "grad"):
        # imb_cals value
        p = np.searchsorted(meg_picks, pick_types(raw.info, meg=key, exclude=()))
        r2 = np.dot(got_imb[p], want_imb[p]) / (
            np.linalg.norm(want_imb[p]) * np.linalg.norm(got_imb[p])
        )
        assert cl[key][0] < r2 <= cl[key][1], f"{key}: {r2:0.3f}"
        # rotation angles
        want_orig_max_angle = want_orig_angles[p].max()
        got_orig_max_angle = got_orig_angles[p].max()
        got_want_max_angle = got_want_angles[p].max()
        if key == "mag":
            assert 8 < want_orig_max_angle < 11, want_orig_max_angle
            assert 1 < got_orig_max_angle < 8, got_orig_max_angle
            assert 8 < got_want_max_angle < 11, got_want_max_angle
        else:
            # Some of these angles are large, but mostly this has to do with
            # processing a very short (one 10-s segment), downsampled (90 Hz)
            # file
            assert gwoma[0] < want_orig_max_angle < gwoma[1]
            assert ggoma[0] < got_orig_max_angle < ggoma[1]
            assert ggwma[0] < got_want_max_angle < ggwma[1]

    kwargs = dict(bad_condition="warning", cross_talk=ctc, coord_frame="meg")
    raw_sss = maxwell_filter(raw, **kwargs)
    raw_sss_mf = maxwell_filter(raw, calibration=cal_mf_fname, **kwargs)
    raw_sss_py = maxwell_filter(raw, calibration=got_cal, **kwargs)
    _assert_shielding(raw_sss, raw, *sfs[0:2])
    _assert_shielding(raw_sss_mf, raw, *sfs[2:4])
    _assert_shielding(raw_sss_py, raw, *sfs[4:6])

    # redoing with given mag data should yield same result
    got_cal_redo, _ = compute_fine_calibration(
        raw, cross_talk=ctc, n_imbalance=1, calibration=got_cal, verbose="debug"
    )
    assert got_cal["ch_names"] == got_cal_redo["ch_names"]
    assert_allclose(got_cal["imb_cals"], got_cal_redo["imb_cals"], atol=5e-5)
    assert_allclose(got_cal["locs"], got_cal_redo["locs"], atol=1e-6)
    assert sum((ic == 1.0).any() for ic in got_cal["imb_cals"]) < 2

    # redoing with 3 imlabance parameters should improve the shielding factor
    grad_subpicks = np.searchsorted(meg_picks, pick_types(raw.info, meg="grad"))
    assert len(grad_subpicks) == 204 and grad_subpicks[0] in (0, 1)
    got_grad_imbs = np.array([got_cal["imb_cals"][pick] for pick in grad_subpicks])
    assert got_grad_imbs.shape == (204, 1)
    got_cal_3, _ = compute_fine_calibration(
        raw, cross_talk=ctc, n_imbalance=3, calibration=got_cal, verbose="debug"
    )
    got_grad_3_imbs = np.array([got_cal_3["imb_cals"][pick] for pick in grad_subpicks])
    assert got_grad_3_imbs.shape == (204, 3)
    corr = np.corrcoef(got_grad_3_imbs[:, 0], got_grad_imbs[:, 0])[0, 1]
    assert cl3[0] < corr < cl3[1]
    raw_sss_py = maxwell_filter(raw, calibration=got_cal_3, **kwargs)
    _assert_shielding(raw_sss_py, raw, *sfs[6:8])


@pytest.mark.parametrize(
    "system",
    [
        pytest.param("kit", marks=[pytest.mark.ultraslowtest]),  # ~6s
        pytest.param("ctf", marks=[td_mark, pytest.mark.ultraslowtest]),  # ~13s
        pytest.param("fil", marks=[td_mark]),  # ~3s
        pytest.param("triux", marks=[td_mark, pytest.mark.slowtest]),  # ~7s
    ],
)
def test_fine_cal_systems(system, tmp_path):
    """Test fine calibration with different systems."""
    int_order = 8
    n_ref = 0
    if system == "kit":
        raw = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
        angle_limit = 170
        err_limit = 500
        n_ref = 3
        corrs = (0.58, 0.61, 0.57)
        sfs = [0.9, 1.1, 2.1, 2.8]
        corr_tol = 0.3
    elif system == "ctf":
        raw = read_raw_ctf(ctf_fname_continuous).crop(0, 1)
        raw.apply_gradient_compensation(0)
        angle_limit = 170
        err_limit = 12600
        n_ref = 28
        corrs = (0.19, 0.41, 0.49)
        sfs = [0.5, 0.7, 0.9, 1.55]
        corr_tol = 0.55
    elif system == "fil":
        raw = read_raw_fil(fil_fname, verbose="error")
        raw.info["bads"] = [f"G2-{a}-{b}" for a in ("MW", "DS", "DT") for b in "YZ"]
        raw.pick("mag", exclude="bads")  # no sensor positions
        raw.crop(1, 2)
        angle_limit = 55
        err_limit = 15
        int_order = 5
        corrs = (0.13, 0.0, 0.12)
        sfs = [4, 5, 125, 155]
        corr_tol = 0.34
    else:
        assert system == "triux", f"Unknown system {system}"
        raw = read_raw_fif(tri_fname)
        angle_limit = 7
        err_limit = 10
        corrs = (-0.13, 0.01, 0.11)
        sfs = [26, 28, 100, 110]
        corr_tol = 0.05
    raw.info["dev_head_t"] = None  # fake empty-room even if it's not
    # avoid line noise and speed up computation
    raw.load_data().resample(50, method="polyphase")
    fc, n = compute_fine_calibration(
        raw,
        angle_limit=angle_limit,
        err_limit=err_limit,
        verbose=True,
    )
    assert n == 1
    # ensure ref sensors not in fine calibration
    ref_picks = pick_types(raw.info, meg=False, ref_meg=True)
    assert len(ref_picks) == n_ref
    for pick in ref_picks:
        assert raw.info["ch_names"][pick] not in fc["ch_names"]
    # write it, read it back, ensure it can be applied
    fname = tmp_path / "fc.dat"
    write_fine_calibration(fname, fc)
    fc_in = read_fine_calibration(fname)
    kwargs = dict(
        coord_frame="meg",
        origin=(0.0, 0.0, 0.0),
        ignore_ref=True,
        regularize=None,
        bad_condition="ignore",
        int_order=int_order,
    )
    raw_sss = maxwell_filter(raw, **kwargs)
    _assert_shielding(raw_sss, raw, *sfs[0:2])
    raw_sss_cal = maxwell_filter(raw, calibration=fc_in, **kwargs)
    _assert_shielding(raw_sss_cal, raw, *sfs[2:4])
    raw_data = raw.get_data("mag").ravel()
    raw_sss_data = raw_sss.get_data("mag").ravel()
    raw_sss_cal_data = raw_sss_cal.get_data("mag").ravel()
    got_corrs = np.corrcoef([raw_data, raw_sss_data, raw_sss_cal_data])
    got_corrs = got_corrs[np.triu_indices(3, 1)]
    assert_allclose(got_corrs, corrs, atol=corr_tol)
    if system == "fil":
        with catch_logging(verbose=True) as log:
            compute_fine_calibration(
                raw.copy().crop(0, 0.12).pick(raw.ch_names[:12]),
                t_window=0.06,  # 2 segments
                angle_limit=angle_limit,
                err_limit=err_limit,
                ext_order=2,
                verbose=True,
            )
        log = log.getvalue()
        assert "(averaging over 2 time intervals)" in log, log
