# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest as pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import BaseRaw, read_raw_nirx, read_raw_snirf
from mne.preprocessing.nirs import optical_density
from mne.utils import _validate_type

fname_nirx = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)
fname_labnirs_multi_wavelength = (
    data_path(download=False) / "SNIRF" / "Labnirs" / "labnirs_3wl_raw_recording.snirf"
)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname,readerfn",
    [(fname_nirx, read_raw_nirx), (fname_labnirs_multi_wavelength, read_raw_snirf)],
)
def test_optical_density(fname, readerfn):
    """Test return type for optical density."""
    if fname.suffix == ".snirf":
        pytest.importorskip("h5py")
    raw_volt = readerfn(fname, preload=False)
    _validate_type(raw_volt, BaseRaw, "raw")

    raw_od = optical_density(raw_volt)
    _validate_type(raw_od, BaseRaw, "raw")

    # Verify data types
    assert set(raw_volt.get_channel_types()) == {"fnirs_cw_amplitude"}
    assert set(raw_od.get_channel_types()) == {"fnirs_od"}

    # Verify that channel names did not change
    for oldname, newname in zip(raw_volt.ch_names, raw_od.ch_names):
        assert oldname == newname

    # Cannot run OD conversion on OD data
    with pytest.raises(RuntimeError, match="on continuous wave"):
        optical_density(raw_od)


@testing.requires_testing_data
def test_optical_density_zeromean():
    """Test that optical density can process zero mean data."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    raw._data[4] -= np.mean(raw._data[4])
    raw._data[4, -1] = 0
    with np.errstate(invalid="raise", divide="raise"):
        with pytest.warns(RuntimeWarning, match="Negative"):
            raw = optical_density(raw)
    assert "fnirs_od" in raw


@testing.requires_testing_data
def test_optical_density_manual():
    """Test optical density on known values."""
    test_tol = 0.01
    raw = read_raw_nirx(fname_nirx, preload=True)
    # log(1) = 0
    raw._data[4] = np.ones(145)
    # log(0.5)/-1 = 0.69
    # log(1.5)/-1 = -0.40
    test_data = np.tile([0.5, 1.5], 73)[:145]
    raw._data[5] = test_data

    od = optical_density(raw)
    assert_allclose(od.get_data([4]), 0.0)
    assert_allclose(od.get_data([5])[0, :2], [0.69, -0.4], atol=test_tol)
