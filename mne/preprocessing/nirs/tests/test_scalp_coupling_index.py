# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
)

fname_nirx_15_0 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_0_recording"
)
fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)
fname_nirx_15_2_short = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname", ([fname_nirx_15_2_short, fname_nirx_15_2, fname_nirx_15_0])
)
@pytest.mark.parametrize("fmt", ("nirx", "fif"))
def test_scalp_coupling_index(fname, fmt, tmp_path):
    """Test converting NIRX files."""
    assert fmt in ("nirx", "fif")
    raw = read_raw_nirx(fname)
    with pytest.raises(RuntimeError, match="Scalp"):
        scalp_coupling_index(raw)

    raw = optical_density(raw)
    sci = scalp_coupling_index(raw)

    # All values should be between -1 and +1
    assert_array_less(sci, 1.0)
    assert_array_less(sci * -1.0, 1.0)

    # Fill in some data with known correlation values
    rng = np.random.RandomState(0)
    new_data = rng.rand(raw._data[0].shape[0])
    # Set first two channels to perfect correlation
    raw._data[0] = new_data
    raw._data[1] = new_data
    # Set next two channels to perfect correlation
    raw._data[2] = new_data
    raw._data[3] = new_data * 0.3  # check scale invariance
    # Set next two channels to anti correlation
    raw._data[4] = new_data
    raw._data[5] = new_data * -1.0
    # Set next two channels to be uncorrelated
    raw._data[6] = new_data
    raw._data[7] = rng.rand(raw._data[0].shape[0])
    # Set next channel to have zero std
    raw._data[8] = 0.0
    raw._data[9] = 1.0
    raw._data[10] = 2.0
    raw._data[11] = 3.0
    # Check values
    sci = scalp_coupling_index(raw)
    assert_allclose(sci[0:6], [1, 1, 1, 1, -1, -1], atol=0.01)
    assert np.abs(sci[6]) < 0.5
    assert np.abs(sci[7]) < 0.5
    assert_allclose(sci[8:12], 0, atol=1e-10)

    # Ensure function errors if wrong type is passed in
    raw = beer_lambert_law(raw, ppf=6)
    with pytest.raises(RuntimeError, match="Scalp"):
        scalp_coupling_index(raw)


@pytest.mark.parametrize("multi_wavelength_raw", [12], indirect=True)
def test_scalp_coupling_index_multi_wavelength(multi_wavelength_raw):
    """Validate SCI min-correlation logic for >=3 wavelengths.

    Similar to test in test_scalp_coupling_index, considers cases
    specific to multi-wavelength data. Uses the `multi_wavelength_raw`
    fixture to generate CW nirs data with the requested number of
    channels (S-D optode pairs), each with 3 wavelengths; in total
    n_channels x 3 data vectors.
    """
    raw = optical_density(multi_wavelength_raw.copy())
    assert len(raw.ch_names) == 12 * 3
    assert raw.ch_names[0] == "S1_D1 700"
    times = np.arange(raw.n_times) / raw.info["sfreq"]
    signal = np.sin(2 * np.pi * 1.0 * times) + 1
    rng = np.random.default_rng()

    # pre-determined expected results
    expected = []
    # group 1: perfect correlation; sci = 1
    raw._data[0] = signal
    raw._data[1] = signal
    raw._data[2] = signal
    expected.extend([1.0] * 3)
    # group 2: scale invariance; sci = 1
    raw._data[3] = signal
    raw._data[4] = signal * 0.3
    raw._data[5] = signal
    expected.extend([1.0] * 3)
    # group 3: anti-correlation; minimum value taken, sci = -1
    raw._data[6] = signal
    raw._data[7] = signal
    raw._data[8] = -signal
    expected.extend([-1.0] * 3)
    # group 4: one zero std channel; minimum value is sci = 0
    raw._data[9] = 0.0
    raw._data[10] = signal
    raw._data[11] = signal
    expected.extend([0.0] * 3)
    # group 5: three zero std channels; all sci = 0
    raw._data[12] = 0.0
    raw._data[13] = 1.0
    raw._data[14] = 2.0
    expected.extend([0.0] * 3)
    # group 6: mixed: 1 signal + 1 negative + 1 random (lowest wins)
    raw._data[15] = signal
    raw._data[16] = rng.random(signal.shape)
    raw._data[17] = -signal
    expected.extend([-1.0] * 3)

    # exact results unknown
    # group 7: 1 uncorrelated signal out of 3; sci < 0.5
    raw._data[18] = signal
    raw._data[19] = rng.random(signal.shape)
    raw._data[20] = signal
    # group 8: 2 uncorrelated signals out of 3; sci < 0.5
    raw._data[21] = rng.random(signal.shape)
    raw._data[22] = rng.random(signal.shape)
    raw._data[23] = signal
    # group 9: 3 uncorrelated signals; sci < 0.5
    raw._data[24] = rng.random(signal.shape)
    raw._data[25] = rng.random(signal.shape)
    raw._data[26] = rng.random(signal.shape)
    # groups 10-12: ordering invariance; all must be the same
    rand1 = rng.random(signal.shape)
    rand2 = rng.random(signal.shape)
    rand3 = rng.random(signal.shape)
    raw._data[27] = rand1
    raw._data[28] = rand2
    raw._data[29] = rand3
    raw._data[30] = rand2
    raw._data[31] = rand1
    raw._data[32] = rand3
    raw._data[33] = rand3
    raw._data[34] = rand1
    raw._data[35] = rand2

    sci = scalp_coupling_index(raw)

    assert_allclose(sci[:18], expected, atol=1e-4)
    for ii in range(18, 27):
        assert np.abs(sci[ii]) < 0.5
    assert_allclose(sci[28:], sci[27], atol=1e-4)
