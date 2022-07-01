import os.path as op

from mne.io import read_raw_eyelink

from mne.datasets.testing import data_path, requires_testing_data

testing_path = data_path(download=False)
fname = op.join(testing_path, 'eyetrack', 'test_eyelink.asc')

raw = read_raw_eyelink(fname, interpolate_missing=True, annotate_missing=True)

print(raw)


####
@requires_testing_data
def test_eyelink_asc():
    """Test reading eyelink asc files."""
    fname = op.join(testing_path, 'eyetrack', 'test_eyelink.asc')

    raw = read_raw_eyelink(fname, interpolate_missing=False)
    assert 'RawEyelink' in repr(raw)

    print(raw)
