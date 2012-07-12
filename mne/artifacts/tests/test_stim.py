import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from ...fiff import Raw
from ...event import read_events
from ..stim import eliminate_stim_artifact

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')


def test_find_ecg():
    """Test eliminate stim artifact"""
    raw = Raw(raw_fname, preload=True)
    events = read_events(event_fname)
    n_events = len(events)
    raw = eliminate_stim_artifact(raw, events, event_id=1, tmin=-0.005,
                                  tmax=0.01, mode='linear')
