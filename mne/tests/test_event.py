import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne import (read_events, write_events, make_fixed_length_events,
                 find_events, fiff)
from mne.utils import _TempDir
from mne.event import define_target_events

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
    write_events(op.join(tempdir, 'events.fif'), events)
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
    raw = fiff.Raw(raw_fname, preload=True)
    events2 = find_events(raw)
    assert_array_almost_equal(events, events2)

    # Test that we can handle consecutive events with no gap
    stim_channel = fiff.pick_channels(raw.info['ch_names'], include='STI 014')
    raw._data[stim_channel, :] = 0
    raw._data[stim_channel, 0] = 1
    raw._data[stim_channel, 10:20] = 5
    raw._data[stim_channel, 20:30] = 6
    raw._data[stim_channel, 30:32] = 5
    raw._data[stim_channel, 40] = 6
    raw._data[stim_channel, -1] = 9
    # Re-reference first_samp to 0 for ease of comparison
    raw.first_samp = 0

    assert_array_equal(find_events(raw),
         [[    0,     0,     1],
          [   10,     0,     5],
          [   40,     0,     6],
          [14399,     0,     9]])
    assert_array_equal(find_events(raw, consecutive=True),
         [[    0,     0,     1],
          [   10,     0,     5],
          [   20,     0,     6],
          [   30,     0,     5],
          [   40,     0,     6],
          [14399,     0,     9]])
    assert_array_equal(find_events(raw, detect='offset'),
         [[    0,     0,     1],
          [   31,     0,     5],
          [   40,     0,     6],
          [14399,     0,     9]])
    assert_array_equal(find_events(raw, detect='offset', consecutive=True),
         [[    0,     0,     1],
          [   19,     0,     5],
          [   29,     0,     6],
          [   31,     0,     5],
          [   40,     0,     6],
          [14399,     0,     9]])
    assert_array_equal(find_events(raw, min_duration=2),
         [[   10,     0,     5]])
    assert_array_equal(find_events(raw, consecutive=True, min_duration=2),
         [[   10,     0,     5],
          [   20,     0,     6],
          [   30,     0,     5]])
    assert_array_equal(find_events(raw, detect='offset', min_duration=2),
         [[   31,     0,     5]])
    assert_array_equal(find_events(raw, detect='offset', consecutive=True,
                                   min_duration=2),
         [[   19,     0,     5],
          [   29,     0,     6],
          [   31,     0,     5]])
    assert_array_equal(find_events(raw, consecutive=True, min_duration=3),
         [[   10,     0,     5],
          [   20,     0,     6]])


def test_make_fixed_length_events():
    """Test making events of a fixed length
    """
    raw = fiff.Raw(raw_fname)
    events = make_fixed_length_events(raw, id=1)
    assert_true(events.shape[1], 3)


def test_define_events():
    """Test defining response events
    """
    events = read_events(fname)
    raw = fiff.Raw(raw_fname)
    events_, _ = define_target_events(events, 5, 32, raw.info['sfreq'],
        .2, 0.7, 42, 99)
    n_target = events[events[:, 2] == 5].shape[0]
    n_miss = events_[events_[:, 2] == 99].shape[0]
    n_target_ = events_[events_[:, 2] == 42].shape[0]

    assert_true(n_target_ == (n_target - n_miss))
