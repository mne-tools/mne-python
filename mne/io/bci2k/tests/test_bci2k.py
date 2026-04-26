# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
from mne.datasets import testing
from mne.io.bci2k.bci2k import (
    _parse_bci2k_header,
    _parse_value_with_unit,
)

data_path = testing.data_path(download=False)
bci2k_fname = data_path / "BCI2k" / "bci2k_test.dat"


@testing.requires_testing_data
def test_read_raw_bci2k():
    """Test reading BCI2000 .dat file."""
    raw = mne.io.read_raw_bci2k(bci2k_fname, preload=True)

    assert raw.info["sfreq"] == 256
    assert raw.info["nchan"] == 3
    assert raw.ch_names == ["EEG1", "EEG2", "STI 014"]
    assert raw.get_channel_types() == ["eeg", "eeg", "stim"]
    assert raw.get_data().shape == (3, raw.n_times)
    assert raw.n_times > 0

    events = mne.find_events(raw, shortest_event=1)
    assert events.ndim == 2
    assert events.shape[1] == 3
    assert "RawBCI2k" in repr(raw)
   
    info_dict = _parse_bci2k_header(bci2k_fname)
    assert info_dict["params"]["SourceChOffset"] == ["0", "0"]
    assert info_dict["params"]["SourceChGain"] == ["0.1muV", "0.1muV"]


def test_parse_value_with_unit():
    """Test numeric token parsing with embedded unit suffixes."""
    volt_scale = {"v": 1.0, "mv": 1e-3, "muv": 1e-6, "uv": 1e-6, "nv": 1e-9}
    assert _parse_value_with_unit("0.1muV", unit_scale=volt_scale) == (0.1, 1e-6)
    assert _parse_value_with_unit("2mV", unit_scale=volt_scale) == (2.0, 1e-3)
    assert _parse_value_with_unit("-3.5µV", unit_scale=volt_scale) == (-3.5, 1e-6)

    freq_scale = {"hz": 1.0, "khz": 1e3}
    value, scale = _parse_value_with_unit("256Hz", unit_scale=freq_scale)
    assert value * scale == 256
    value, scale = _parse_value_with_unit("0.5kHz", unit_scale=freq_scale)
    assert value * scale == 500
