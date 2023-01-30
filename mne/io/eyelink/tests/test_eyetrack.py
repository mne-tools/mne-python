import pytest
from mne.io import read_raw_eyelink
from mne.io.constants import FIFF
from mne.datasets.testing import data_path, requires_testing_data


testing_path = data_path(download=False)
fname = testing_path / 'eyetrack' / 'test_eyelink.asc'


@requires_testing_data
@pytest.mark.parametrize('fname, create_annotations, find_overlaps',
                         [(fname, False, False),
                          (fname, True, False),
                          (fname, True, True)])
def test_eyelink(fname, create_annotations, find_overlaps):
    """Test reading eyelink asc files."""
    raw = read_raw_eyelink(fname, create_annotations, find_overlaps)

    # First, tests that shouldn't change based on function arguments
    assert raw.info['sfreq'] == 500  # True for this file
    assert len(raw.info['ch_names']) == 6
    assert raw.info['chs'][0]['kind'] == FIFF.FIFFV_EYETRACK_CH
    assert 'RawEyelink' in repr(raw)

    # TODO: test some annotation values for accuracy.
