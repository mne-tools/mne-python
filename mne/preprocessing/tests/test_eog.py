import os.path as op
from nose.tools import assert_true

from mne.io import read_raw_fif
from mne.preprocessing.eog import find_eog_events

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_eog():
    """Test find EOG peaks."""
    raw = read_raw_fif(raw_fname)
    events = find_eog_events(raw)
    n_events = len(events)
    assert_true(n_events == 4)
