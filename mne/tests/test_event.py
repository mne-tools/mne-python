import os.path as op

from numpy.testing import assert_array_almost_equal

import mne


fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-eve.fif')

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')


def test_io_events():
    """Test IO for events
    """
    events = mne.read_events(fname)
    mne.write_events('events.fif', events)
    events2 = mne.read_events('events.fif')
    assert_array_almost_equal(events, events2)


def test_find_events():
    """Test find events in raw file
    """
    events = mne.read_events(fname)
    raw = mne.fiff.Raw(raw_fname)
    events2 = mne.find_events(raw)
    assert_array_almost_equal(events, events2)
