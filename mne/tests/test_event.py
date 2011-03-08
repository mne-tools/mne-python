import os.path as op

from numpy.testing import assert_array_almost_equal

import mne


fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-eve.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    events = mne.read_events(fname)
    mne.write_events('events.fif', events)
    events2 = mne.read_events(fname)
    assert_array_almost_equal(events, events2)
