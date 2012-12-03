import os.path as op
from nose.tools import assert_true

from mne.fiff import Raw
from mne.preprocessing.eog import find_eog_events

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test_proj.fif')


def test_find_eog():
    """Test find EOG peaks"""
    raw = Raw(raw_fname)
    events = find_eog_events(raw)
    n_events = len(events)
    assert_true(n_events == 4)
