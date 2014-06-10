import os.path as op
import os

from nose.tools import assert_true
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_raises)
import warnings

from mne import (read_events, write_events, make_fixed_length_events,
                 find_events, find_stim_steps, io, pick_channels)
from mne.utils import _TempDir
from mne.event import define_target_events, merge_events

warnings.simplefilter('always')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname = op.join(base_dir, 'test-eve.fif')
fname_gz = op.join(base_dir, 'test-eve.fif.gz')
fname_1 = op.join(base_dir, 'test-1-eve.fif')
fname_txt = op.join(base_dir, 'test-eve.eve')
fname_txt_1 = op.join(base_dir, 'test-eve-1.eve')

# using mne_process_raw --raw test_raw.fif --eventsout test-mpr-eve.eve:
fname_txt_mpr = op.join(base_dir, 'test-mpr-eve.eve')
fname_old_txt = op.join(base_dir, 'test-eve-old-style.eve')
raw_fname = op.join(base_dir, 'test_raw.fif')

tempdir = _TempDir()


def test_add_events():
    """Test adding events to a Raw file"""
    # need preload
    raw = io.Raw(raw_fname, preload=False)
    events = np.array([[raw.first_samp, 0, 1]])
    assert_raises(RuntimeError, raw.add_events, events, 'STI 014')
    raw = io.Raw(raw_fname, preload=True)
    orig_events = find_events(raw, 'STI 014')
    # add some events
    events = np.array([raw.first_samp, 0, 1])
    assert_raises(ValueError, raw.add_events, events, 'STI 014')  # bad shape
    events[0] = raw.first_samp + raw.n_times + 1
    events = events[np.newaxis, :]
    assert_raises(ValueError, raw.add_events, events, 'STI 014')  # bad time
    events[0, 0] = raw.first_samp - 1
    assert_raises(ValueError, raw.add_events, events, 'STI 014')  # bad time
    events[0, 0] = raw.first_samp + 1  # can't actually be first_samp
    assert_raises(ValueError, raw.add_events, events, 'STI FOO')
    raw.add_events(events, 'STI 014')
    new_events = find_events(raw, 'STI 014')
    assert_array_equal(new_events, np.concatenate((events, orig_events)))


def test_merge_events():
    """Test event merging
    """
    events = read_events(fname)  # Use as the gold standard
    merges = [1, 2, 3, 4]
    events_out = merge_events(events, merges, 1234)
    events_out2 = events.copy()
    for m in merges:
        assert_true(not np.any(events_out[:, 2] == m))
        events_out2[events[:, 2] == m, 2] = 1234
    assert_array_equal(events_out, events_out2)
    # test non-replacement functionality, should be sorted union of orig & new
    events_out2 = merge_events(events, merges, 1234, False)
    events_out = np.concatenate((events_out, events))
    events_out = events_out[np.argsort(events_out[:, 0])]
    assert_array_equal(events_out, events_out2)


def test_io_events():
    """Test IO for events
    """
    # Test binary fif IO
    events = read_events(fname)  # Use as the gold standard
    write_events(op.join(tempdir, 'events-eve.fif'), events)
    events2 = read_events(op.join(tempdir, 'events-eve.fif'))
    assert_array_almost_equal(events, events2)

    # Test binary fif.gz IO
    events2 = read_events(fname_gz)  # Use as the gold standard
    assert_array_almost_equal(events, events2)
    write_events(op.join(tempdir, 'events-eve.fif.gz'), events2)
    events2 = read_events(op.join(tempdir, 'events-eve.fif.gz'))
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
    a = read_events(op.join(tempdir, 'events-eve.fif'), include=1)
    b = read_events(op.join(tempdir, 'events-eve.fif'), include=[1])
    c = read_events(op.join(tempdir, 'events-eve.fif'), exclude=[2, 3, 4, 5, 32])
    d = read_events(op.join(tempdir, 'events-eve.fif'), include=1, exclude=[2, 3])
    assert_array_equal(a, b)
    assert_array_equal(a, c)
    assert_array_equal(a, d)

    # Test binary file IO for 1 event
    events = read_events(fname_1)  # Use as the new gold standard
    write_events(op.join(tempdir, 'events-eve.fif'), events)
    events2 = read_events(op.join(tempdir, 'events-eve.fif'))
    assert_array_almost_equal(events, events2)

    # Test text file IO for 1 event
    write_events(op.join(tempdir, 'events.eve'), events)
    events2 = read_events(op.join(tempdir, 'events.eve'))
    assert_array_almost_equal(events, events2)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fname2 = op.join(tempdir, 'test-bad-name.fif')
        write_events(fname2, events)
        read_events(fname2)
    assert_true(len(w) == 2)


def test_find_events():
    """Test find events in raw file
    """
    events = read_events(fname)
    raw = io.Raw(raw_fname, preload=True)
    # let's test the defaulting behavior while we're at it
    extra_ends = ['', '_1']
    orig_envs = [os.getenv('MNE_STIM_CHANNEL%s' % s) for s in extra_ends]
    os.environ['MNE_STIM_CHANNEL'] = 'STI 014'
    if 'MNE_STIM_CHANNEL_1' in os.environ:
        del os.environ['MNE_STIM_CHANNEL_1']
    events2 = find_events(raw)
    assert_array_almost_equal(events, events2)

    # Reset some data for ease of comparison
    raw.first_samp = 0
    raw.info['sfreq'] = 1000

    stim_channel = 'STI 014'
    stim_channel_idx = pick_channels(raw.info['ch_names'],
                                      include=stim_channel)

    # test empty events channel
    raw._data[stim_channel_idx, :] = 0
    assert_array_equal(find_events(raw), np.empty((0, 3), dtype='int32'))

    raw._data[stim_channel_idx, :4] = 1
    assert_array_equal(find_events(raw), np.empty((0, 3), dtype='int32'))

    raw._data[stim_channel_idx, -1:] = 9
    assert_array_equal(find_events(raw), [[14399, 0, 9]])

    # Test that we can handle consecutive events with no gap
    raw._data[stim_channel_idx, 10:20] = 5
    raw._data[stim_channel_idx, 20:30] = 6
    raw._data[stim_channel_idx, 30:32] = 5
    raw._data[stim_channel_idx, 40] = 6

    assert_array_equal(find_events(raw, consecutive=False),
                       [[10, 0, 5],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_array_equal(find_events(raw, consecutive=True),
                       [[10, 0, 5],
                        [20, 5, 6],
                        [30, 6, 5],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_array_equal(find_events(raw),
                       [[10, 0, 5],
                        [20, 5, 6],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_array_equal(find_events(raw, output='offset', consecutive=False),
                       [[31, 0, 5],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_array_equal(find_events(raw, output='offset', consecutive=True),
                       [[19, 6, 5],
                        [29, 5, 6],
                        [31, 0, 5],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_raises(ValueError,find_events,raw, output='step', consecutive=True)
    assert_array_equal(find_events(raw, output='step', consecutive=True,
                                   shortest_event=1),
                       [[10, 0, 5],
                        [20, 5, 6],
                        [30, 6, 5],
                        [32, 5, 0],
                        [40, 0, 6],
                        [41, 6, 0],
                        [14399, 0, 9],
                        [14400, 9, 0]])
    assert_array_equal(find_events(raw, output='offset'),
                       [[19, 6, 5],
                        [31, 0, 6],
                        [40, 0, 6],
                        [14399, 0, 9]])
    assert_array_equal(find_events(raw, consecutive=False, min_duration=0.002),
                       [[10, 0, 5]])
    assert_array_equal(find_events(raw, consecutive=True, min_duration=0.002),
                       [[10, 0, 5],
                        [20, 5, 6],
                        [30, 6, 5]])
    assert_array_equal(find_events(raw, output='offset', consecutive=False,
                                   min_duration=0.002),
                       [[31, 0, 5]])
    assert_array_equal(find_events(raw, output='offset', consecutive=True,
                                   min_duration=0.002),
                       [[19, 6, 5],
                        [29, 5, 6],
                        [31, 0, 5]])
    assert_array_equal(find_events(raw, consecutive=True, min_duration=0.003),
                       [[10, 0, 5],
                        [20, 5, 6]])

    # test find_stim_steps merge parameter
    raw._data[stim_channel_idx, :] = 0
    raw._data[stim_channel_idx, 0] = 1
    raw._data[stim_channel_idx, 10] = 4
    raw._data[stim_channel_idx, 11:20] = 5
    assert_array_equal(find_stim_steps(raw, pad_start=0, merge=0,
                                       stim_channel=stim_channel),
                       [[0, 0, 1],
                        [1, 1, 0],
                        [10, 0, 4],
                        [11, 4, 5],
                        [20, 5, 0]])
    assert_array_equal(find_stim_steps(raw, merge=-1,
                                       stim_channel=stim_channel),
                       [[1, 1, 0],
                        [10, 0, 5],
                        [20, 5, 0]])
    assert_array_equal(find_stim_steps(raw, merge=1,
                                       stim_channel=stim_channel),
                       [[1, 1, 0],
                        [11, 0, 5],
                        [20, 5, 0]])

    # put back the env vars we trampled on
    for s, o in zip(extra_ends, orig_envs):
        if o is not None:
            os.environ['MNE_STIM_CHANNEL%s' % s] = o


def test_make_fixed_length_events():
    """Test making events of a fixed length
    """
    raw = io.Raw(raw_fname)
    events = make_fixed_length_events(raw, id=1)
    assert_true(events.shape[1], 3)


def test_define_events():
    """Test defining response events
    """
    events = read_events(fname)
    raw = io.Raw(raw_fname)
    events_, _ = define_target_events(events, 5, 32, raw.info['sfreq'],
                                      .2, 0.7, 42, 99)
    n_target = events[events[:, 2] == 5].shape[0]
    n_miss = events_[events_[:, 2] == 99].shape[0]
    n_target_ = events_[events_[:, 2] == 42].shape[0]

    assert_true(n_target_ == (n_target - n_miss))
