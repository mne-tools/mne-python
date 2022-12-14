"""Test generic read_raw function."""

# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD-3-Clause

from pathlib import Path

import pytest

from mne.io import read_raw
from mne.datasets import testing


base = Path(__file__).parent.parent
test_base = Path(testing.data_path(download=False))


@pytest.mark.parametrize('fname', ['x.xxx', 'x'])
def test_read_raw_unsupported_single(fname):
    """Test handling of unsupported file types."""
    with pytest.raises(ValueError, match='Unsupported file type'):
        read_raw(fname)


@pytest.mark.parametrize('fname', ['x.bin'])
def test_read_raw_unsupported_multi(fname, tmp_path):
    """Test handling of supported file types but with bad data."""
    fname = tmp_path / fname
    fname.write_text('')
    with pytest.raises(RuntimeError, match='Could not read.*using any'):
        read_raw(fname)


@pytest.mark.parametrize('fname', ['x.vmrk', 'x.eeg'])
def test_read_raw_suggested(fname):
    """Test handling of unsupported file types with suggested alternatives."""
    with pytest.raises(ValueError, match='Try reading'):
        read_raw(fname)


_testing_mark = testing._pytest_mark()


@pytest.mark.parametrize('fname', [
    base / 'edf/tests/data/test.edf',
    base / 'edf/tests/data/test.bdf',
    base / 'brainvision/tests/data/test.vhdr',
    base / 'kit/tests/data/test.sqd',
    pytest.param(test_base / 'KIT' / 'data_berlin.con', marks=_testing_mark),
    pytest.param(
        test_base / 'ARTEMIS123' /
        'Artemis_Data_2017-04-14-10h-38m-59s_Phantom_1k_HPI_1s.bin',
        marks=_testing_mark),
    pytest.param(
        test_base / 'FIL' /
        'sub-noise_ses-001_task-noise220622_run-001_meg.bin',
        marks=(
            _testing_mark,
            pytest.mark.filterwarnings(
                'ignore:.*problems later!:RuntimeWarning'))),
])
def test_read_raw_supported(fname):
    """Test supported file types."""
    read_raw(fname)
    read_raw(fname, verbose=False)
    raw = read_raw(fname, preload=True)
    assert "data loaded" in str(raw)
