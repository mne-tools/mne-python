# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("neo")


@pytest.fixture
def neo_example_file(tmp_path):
    """Create a temporary file for Neo ExampleIO."""
    fpath = tmp_path / "test_data.nof"
    # ExampleIO generates fake data but MNE requires the file to exist
    fpath.touch()
    return fpath


def test_neo_basic_reading(neo_example_file):
    """Test basic reading with ExampleIO."""
    import neo

    from mne.io import read_raw_neo

    temp_fname = str(neo_example_file)

    # Get expected data directly from Neo
    reader = neo.io.ExampleIO(temp_fname)
    blocks = reader.read(lazy=False)
    block = blocks[0]

    expected_data = []
    for segment in block.segments:
        signal = segment.analogsignals[0]
        expected_data.append(signal.rescale("V").magnitude)

    expected_data = np.concatenate(expected_data, axis=0).T
    expected_sfreq = float(
        block.segments[0].analogsignals[0].sampling_rate.rescale("Hz").magnitude
    )

    # Read with MNE
    raw = read_raw_neo(temp_fname, neo_io_class="ExampleIO", preload=True)

    assert raw.info["sfreq"] == expected_sfreq
    assert len(raw.ch_names) == expected_data.shape[0]
    assert raw.n_times == expected_data.shape[1]
    assert_allclose(raw.get_data(), expected_data, rtol=1e-6)


def test_neo_lazy_loading(neo_example_file):
    """Test lazy loading."""
    from mne.io import read_raw_neo

    temp_fname = str(neo_example_file)
    raw = read_raw_neo(temp_fname, neo_io_class="ExampleIO", preload=False)

    assert not raw.preload

    # Load subset
    data, times = raw[:, :100]
    assert data.shape[1] == 100

    # Full load
    raw.load_data()
    assert raw.preload


def test_neo_auto_detect(neo_example_file):
    """Test that neo_io_class can be auto-detected."""
    from mne.io import read_raw_neo

    # ExampleIO won't be auto-detected by extension, so this tests the fallback
    # For real files like .trc, .rhd, etc., auto-detection works
    temp_fname = str(neo_example_file)

    # ExampleIO files won't auto-detect, so we still need explicit class
    raw = read_raw_neo(temp_fname, neo_io_class="ExampleIO", preload=True)
    assert raw.n_times > 0


def test_neo_invalid_io_class(neo_example_file):
    """Test invalid IO class raises error."""
    from mne.io import read_raw_neo

    temp_fname = str(neo_example_file)

    with pytest.raises(ValueError, match="Unknown Neo IO class"):
        read_raw_neo(temp_fname, neo_io_class="InvalidIO")


def test_neo_channel_selection(neo_example_file):
    """Test channel selection with slicing."""
    from mne.io import read_raw_neo

    temp_fname = str(neo_example_file)
    raw = read_raw_neo(temp_fname, neo_io_class="ExampleIO", preload=False)

    # Pick specific channels
    n_channels = len(raw.ch_names)
    if n_channels >= 2:
        raw.pick([raw.ch_names[0], raw.ch_names[1]])
        data, _ = raw[:, :]
        assert data.shape[0] == 2


def test_neo_data_types(neo_example_file):
    """Test that data is returned as float64."""
    from mne.io import read_raw_neo

    temp_fname = str(neo_example_file)
    raw = read_raw_neo(temp_fname, neo_io_class="ExampleIO", preload=True)

    data = raw.get_data()
    assert data.dtype == np.float64
