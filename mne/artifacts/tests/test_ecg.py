import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from ...fiff import Raw
from ..ecg import find_ecg_events

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test_proj.fif')


def test_find_ecg():
    """Test find ECG peaks"""
    raw = Raw(raw_fname)
    events, ch_ECG, average_pulse = find_ecg_events(raw, event_id=999,
                                                    ch_name='MEG 1531')
    n_events = len(events)
    _, times = raw[0, :]
    assert_true(55 < average_pulse < 60)
