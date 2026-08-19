# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (
    _validate_nirs_info,
    motion_correct_pca,
    optical_density,
    pca,
)

fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)


def _make_tinc(n_times, bad_start, bad_stop):
    """Return a per-time boolean mask (True=clean, False=motion)."""
    tInc = np.ones(n_times, dtype=bool)
    tInc[bad_start:bad_stop] = False
    return tInc


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_pca_removes_shared_artifact(fname):
    """Test PCA correction reduces a correlated motion artefact."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]

    # Inject a correlated step shift in the middle of the recording
    mid = n_times // 2
    bad_start, bad_stop = mid - 10, mid + 10

    # Save clean signal for comparison
    original = raw_od._data[picks[0]].copy()

    max_shift = np.max(np.abs(np.diff(raw_od._data[picks[0]])))
    shift_amp = 20 * max_shift

    for pick in picks:
        raw_od._data[pick, bad_start:bad_stop] += shift_amp

    tInc = _make_tinc(n_times, bad_start, bad_stop)
    raw_od_corr, svs, nSV_ret = motion_correct_pca(raw_od, tInc=tInc, nSV=0.97)

    # Corrected signal should be closer to original than corrupted signal
    mse_before = np.mean((raw_od._data[picks[0]] - original) ** 2)
    mse_after = np.mean((raw_od_corr._data[picks[0]] - original) ** 2)
    assert mse_after < mse_before


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_pca_svs_sum_to_one(fname):
    """Test returned singular values are normalised and sum to 1."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    n_times = raw_od._data.shape[1]

    tInc = _make_tinc(n_times, n_times // 2 - 5, n_times // 2 + 5)
    _, svs, _ = motion_correct_pca(raw_od, tInc=tInc, nSV=0.97)

    assert_allclose(np.sum(svs), 1.0, rtol=1e-6)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_pca_integer_nsv(fname):
    """Test integer nSV removes exactly that many components."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    n_times = raw_od._data.shape[1]

    tInc = _make_tinc(n_times, n_times // 3, n_times // 3 + 20)
    _, _, nSV_ret = motion_correct_pca(raw_od, tInc=tInc, nSV=2)
    assert nSV_ret == 2


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_pca_returns_copy(fname):
    """Test PCA correction does not modify the input Raw in place."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    n_times = raw_od._data.shape[1]
    original = raw_od._data[picks[0]].copy()

    tInc = _make_tinc(n_times, 10, 30)
    _, _, _ = motion_correct_pca(raw_od, tInc=tInc)
    assert_allclose(raw_od._data[picks[0]], original)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_pca_all_good_raises(fname):
    """Test PCA correction raises when tInc has no artefact samples."""
    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    n_times = raw_od._data.shape[1]

    tInc = np.ones(n_times, dtype=bool)  # all clean – no motion to correct
    with pytest.raises(ValueError, match="No motion-artifact samples"):
        motion_correct_pca(raw_od, tInc=tInc)


def test_pca_alias():
    """Test pca is an alias for motion_correct_pca."""
    assert pca is motion_correct_pca


def test_motion_correct_pca_wrong_type():
    """Test passing a non-Raw object raises TypeError."""
    with pytest.raises(TypeError):
        motion_correct_pca(np.zeros((10, 100)), tInc=np.ones(100, dtype=bool))
