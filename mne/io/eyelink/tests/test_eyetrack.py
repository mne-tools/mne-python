import pytest
from mne.io import read_raw_eyelink
from mne.datasets.testing import data_path, requires_testing_data

testing_path = data_path(download=False)
eyelink_fname = fname = testing_path / 'eyetrack' / 'test_eyelink.asc'


@requires_testing_data
@pytest.mark.parametrize('fname, read_eye_events, interpolate_missing, '
                         'annotate_missing',
                         [(eyelink_fname, False, False, False),
                          (eyelink_fname, True, False, False)])
def test_eyelink(fname, read_eye_events, interpolate_missing,
                 annotate_missing):
    """Test reading eyelink asc files."""

    raw = read_raw_eyelink(fname, read_eye_events, interpolate_missing,
                           annotate_missing)
    assert 'RawEyelink' in repr(raw)

    # TODO: test some annotation values for accuracy.
