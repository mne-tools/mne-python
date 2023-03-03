import numpy as np
import pytest
from numpy.random import default_rng

from mne import concatenate_raws, create_info, make_fixed_length_epochs
from mne.io import RawArray


def _create_toy_data(n_channels=3, sfreq=250, seed=None):
    rng = default_rng(seed)
    data = rng.standard_normal(size=(n_channels, 50 * sfreq)) * 5e-6
    info = create_info(n_channels, sfreq, "eeg")
    return RawArray(data, info)


def test_concatenate_raws():
    """Test concatenation of raw instances."""
    raw0 = _create_toy_data()
    raw1 = _create_toy_data()

    # Test bad channel order
    raw0.info["bads"] = ["0", "1"]
    raw1.info["bads"] = ["1", "0"]

    # raw0 is modified in-place and therefore copied
    raw_concat = concatenate_raws([raw0.copy(), raw1])

    # Check data are equal
    data_concat = np.concatenate([raw0.get_data(), raw1.get_data()], 1)
    assert np.all(raw_concat.get_data() == data_concat)

    # Check bad channels
    assert set(raw_concat.info["bads"]) == {"0", "1"}

    # Bad channel mismatch raises
    raw2 = raw1.copy()
    raw2.info["bads"] = ["0", "2"]
    with pytest.raises(ValueError):
        concatenate_raws([raw0, raw2])

    # Type mismatch raises
    epochs1 = make_fixed_length_epochs(raw1)
    with pytest.raises(ValueError):
        concatenate_raws([raw0, epochs1])

    # Sample rate mismatch
    raw3 = _create_toy_data(sfreq=500)
    with pytest.raises(ValueError):
        concatenate_raws([raw0, raw3])

    # Number of channels mismatch
    raw4 = _create_toy_data(n_channels=4)
    with pytest.raises(ValueError):
        concatenate_raws([raw0, raw4])
