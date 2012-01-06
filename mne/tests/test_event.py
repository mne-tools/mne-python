import os.path as op

from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import read_events, write_events, find_events
from .. import fiff


fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-eve.fif')

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')


def test_io_events():
    """Test IO for events
    """
    events = read_events(fname)
    write_events('events.fif', events)
    events2 = read_events('events.fif')
    assert_array_almost_equal(events, events2)

    a = read_events('events.fif', include=1)
    b = read_events('events.fif', include=[1])
    c = read_events('events.fif', exclude=[2, 3, 4, 5, 32])
    d = read_events('events.fif', include=1, exclude=[2, 3])
    assert_array_equal(a, b)
    assert_array_equal(a, c)
    assert_array_equal(a, d)


def test_find_events():
    """Test find events in raw file
    """
    events = read_events(fname)
    raw = fiff.Raw(raw_fname)
    events2 = find_events(raw)
    assert_array_almost_equal(events, events2)
