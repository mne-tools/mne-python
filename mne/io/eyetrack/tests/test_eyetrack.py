from mne.io import read_raw_eyelink

from mne.datasets.testing import data_path, requires_testing_data

fname = '/Users/dominik.welke/Work/11_datasets/random_eyetrack/edf/' \
        'sub-00_task-aeAHA_eye_mini.asc'

raw = read_raw_eyelink(fname)

print(raw)


####
@requires_testing_data
def test_eyelink_asc():
    """Test reading eyelink asc files."""
    fname = '/Users/dominik.welke/Work/11_datasets/random_eyetrack/edf/' \
            'sub-00_task-aeAHA_eye_mini.asc'

    raw = read_raw_eyelink(fname)

    print(raw)



