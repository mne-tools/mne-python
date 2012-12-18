import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import (read_events, write_events, make_fixed_length_events,
                 find_events, fiff)
from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
fname = op.join(base_dir, 'test-eve.fif')
fname_gz = op.join(base_dir, 'test-eve.fif.gz')
fname_1 = op.join(base_dir, 'test-eve-1.fif')
fname_txt = op.join(base_dir, 'test-eve.eve')
fname_txt_1 = op.join(base_dir, 'test-eve-1.eve')

# using mne_process_raw --raw test_raw.fif --eventsout test-mpr-eve.eve:
fname_txt_mpr = op.join(base_dir, 'test-mpr-eve.eve')
fname_old_txt = op.join(base_dir, 'test-eve-old-style.eve')
raw_fname = op.join(base_dir, 'test_raw.fif')

tempdir = _TempDir()


def test_io_events():
    """Test IO for events
    """
    # Test binary fif IO
    events = read_events(fname)  # Use as the gold standard
    write_events(op.join(tempdir,'events.fif'), events)
    events2 = read_events(op.join(tempdir, 'events.fif'))
    assert_array_almost_equal(events, events2)

    # Test binary fif.gz IO
    events2 = read_events(fname_gz)  # Use as the gold standard
    assert_array_almost_equal(events, events2)
    write_events(op.join(tempdir, 'events.fif.gz'), events2)
    events2 = read_events(op.join(tempdir, 'events.fif.gz'))
    assert_array_almost_equal(events, events2)

    # Test new format text file IO
    write_events(op.join(tempdir, 'events.eve'), events)
    events2 = read_events(op.join(tempdir, 'events.eve'))
    assert_array_almost_equal(events, events2)
    events2 = read_events(fname_txt_mpr)
    assert_array_almost_equal(events, events2)

    # Test old format text file IO
    events2 = read_events(fname_old_txt)
    assert_array_almost_equal(events, events2)
    write_events(op.join(tempdir, 'events.eve'), events)
    events2 = read_events(op.join(tempdir, 'events.eve'))
    assert_array_almost_equal(events, events2)

    # Test event selection
    a = read_events(op.join(tempdir, 'events.fif'), include=1)
    b = read_events(op.join(tempdir, 'events.fif'), include=[1])
    c = read_events(op.join(tempdir, 'events.fif'), exclude=[2, 3, 4, 5, 32])
    d = read_events(op.join(tempdir, 'events.fif'), include=1, exclude=[2, 3])
    assert_array_equal(a, b)
    assert_array_equal(a, c)
    assert_array_equal(a, d)

    # Test binary file IO for 1 event
    events = read_events(fname_1)  # Use as the new gold standard
    write_events(op.join(tempdir, 'events.fif'), events)
    events2 = read_events(op.join(tempdir, 'events.fif'))
    assert_array_almost_equal(events, events2)

    # Test text file IO for 1 event
    write_events(op.join(tempdir, 'events.eve'), events)
    events2 = read_events(op.join(tempdir, 'events.eve'))
    assert_array_almost_equal(events, events2)


def test_find_events():
    """Test find events in raw file
    """
    events = read_events(fname)
    raw = fiff.Raw(raw_fname)
    events2 = find_events(raw)
    assert_array_almost_equal(events, events2)


def test_make_fixed_length_events():
    """Test making events of a fixed length
    """
    raw = fiff.Raw(raw_fname)
    events = make_fixed_length_events(raw, id=1)
    assert_true(events.shape[1], 3)
