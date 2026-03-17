# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import mne
from mne.datasets import testing

data_path = testing.data_path(download=False)
bci2k_fname = data_path / "BCI2k" / "bci2k_test.dat"


@testing.requires_testing_data
def test_read_raw_bci2k():
    """Test reading BCI2000 .dat file."""
    raw = mne.io.read_raw_bci2k(bci2k_fname, preload=True)

    assert raw.info["sfreq"] == 256
    assert raw.info["nchan"] == 3

    assert raw.ch_names == ["EEG1", "EEG2", "STI 014"]

    ch_types = raw.get_channel_types()
    assert ch_types == ["eeg", "eeg", "stim"]

    data = raw.get_data()
    assert raw.get_data().shape == (3, raw.n_times)

    assert raw.n_times > 0

    events = mne.find_events(raw, shortest_event=1)

    assert events.ndim == 2
    assert events.shape[1] == 3
    assert "RawBCI2k" in repr(raw)
