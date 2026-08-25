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
    beer_lambert_law,
    motion_correct_wavelet,
    optical_density,
    wavelet,
)

fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_wavelet_reduces_spikes_od(fname):
    """Test wavelet correction attenuates spike artefacts in OD data."""
    pytest.importorskip("pywt")

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)

    sig = raw_od._data[picks[0]]
    spike_amp = 20 * np.std(np.diff(sig))

    # Inject isolated single-sample spikes
    n_times = raw_od._data.shape[1]
    raw_od._data[picks[0], n_times // 4] += spike_amp
    raw_od._data[picks[0], n_times // 2] -= spike_amp

    spike_before = np.max(np.abs(np.diff(raw_od._data[picks[0]])))

    raw_od_corr = motion_correct_wavelet(raw_od, iqr=1.5)

    spike_after = np.max(np.abs(np.diff(raw_od_corr._data[picks[0]])))
    assert spike_after < spike_before


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_wavelet_reduces_spikes_hb(fname):
    """Test wavelet correction works on haemoglobin concentration data."""
    pytest.importorskip("pywt")

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    raw_hb = beer_lambert_law(raw_od)
    picks = _validate_nirs_info(raw_hb.info)

    spike_amp = 20 * np.std(np.diff(raw_hb._data[picks[0]]))
    n_times = raw_hb._data.shape[1]
    raw_hb._data[picks[0], n_times // 4] += spike_amp
    raw_hb._data[picks[0], n_times // 2] -= spike_amp

    spike_before = np.max(np.abs(np.diff(raw_hb._data[picks[0]])))
    raw_hb_corr = motion_correct_wavelet(raw_hb, iqr=1.5)
    spike_after = np.max(np.abs(np.diff(raw_hb_corr._data[picks[0]])))
    assert spike_after < spike_before


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_wavelet_negative_iqr_passthrough(fname):
    """Test iqr < 0 returns data unchanged."""
    pytest.importorskip("pywt")

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    original = raw_od._data[picks[0]].copy()

    raw_od_corr = motion_correct_wavelet(raw_od, iqr=-1)
    assert_allclose(raw_od_corr._data[picks[0]], original)


@testing.requires_testing_data
@pytest.mark.parametrize("fname", ([fname_nirx_15_2]))
def test_motion_correct_wavelet_returns_copy(fname):
    """Test wavelet correction does not modify the input Raw in place."""
    pytest.importorskip("pywt")

    raw = read_raw_nirx(fname)
    raw_od = optical_density(raw)
    picks = _validate_nirs_info(raw_od.info)
    original = raw_od._data[picks[0]].copy()

    _ = motion_correct_wavelet(raw_od)
    assert_allclose(raw_od._data[picks[0]], original)


def test_wavelet_alias():
    """Test wavelet is an alias for motion_correct_wavelet."""
    assert wavelet is motion_correct_wavelet


def test_motion_correct_wavelet_wrong_type():
    """Test passing a non-Raw object raises TypeError."""
    pytest.importorskip("pywt")
    with pytest.raises(TypeError):
        motion_correct_wavelet(np.zeros((10, 100)))
