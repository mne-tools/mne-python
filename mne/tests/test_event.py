import os.path as op
import os

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
import warnings

from mne import (read_events, write_events, make_fixed_length_events,
                 find_events, pick_events, find_stim_steps, pick_channels,
                 read_evokeds, Epochs)
from mne.io import read_raw_fif
from mne.tests.common import assert_naming
from mne.utils import _TempDir, run_tests_if_main
from mne.event import define_target_events, merge_events, AcqParserFIF
from mne.datasets import testing

warnings.simplefilter('always')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname = op.join(base_dir, 'test-eve.fif')
fname_gz = op.join(base_dir, 'test-eve.fif.gz')
fname_1 = op.join(base_dir, 'test-1-eve.fif')
fname_txt = op.join(base_dir, 'test-eve.eve')
fname_txt_1 = op.join(base_dir, 'test-eve-1.eve')

# for testing Elekta averager
elekta_base_dir = op.join(testing.data_path(download=False), 'misc')
fname_raw_elekta = op.join(elekta_base_dir, 'test_elekta_3ch_raw.fif')
fname_ave_elekta = op.join(elekta_base_dir, 'test_elekta-ave.fif')

# using mne_process_raw --raw test_raw.fif --eventsout test-mpr-eve.eve:
fname_txt_mpr = op.join(base_dir, 'test-mpr-eve.eve')
fname_old_txt = op.join(base_dir, 'test-eve-old-style.eve')
raw_fname = op.join(base_dir, 'test_raw.fif')


def test_fix_stim():
    """Test fixing stim STI016 for Neuromag."""
    raw = read_raw_fif(raw_fname, preload=True)
    # 32768 (016) + 3 (002+001) bits gets incorrectly coded during acquisition
    raw._data[raw.ch_names.index('STI 014'), :3] = [0, -32765, 0]
    with warnings.catch_warnings(record=True) as w:
        events = find_events(raw, 'STI 014')
    assert_true(len(w) >= 1)
    assert_true(any('STI016' in str(ww.message) for ww in w))
    assert_array_equal(events[0], [raw.first_samp + 1, 0, 32765])
    events = find_events(raw, 'STI 014', uint_cast=True)
    assert_array_equal(events[0], [raw.first_samp + 1, 0, 32771])


def test_add_events():
    """Test adding events to a Raw file."""
    # need preload
    raw = read_raw_fif(raw_fname)
    events = np.array([[raw.first_samp, 0, 1]])
    assert_raises(RuntimeError, raw.add_events, events, 'STI 014')
    raw = read_raw_fif(raw_fname, preload=True)
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
    """Test event merging."""
    events_orig = [[1, 0, 1], [3, 0, 2], [10, 0, 3], [20, 0, 4]]

    events_replacement = \
        [[1, 0, 12],
         [3, 0, 12],
         [10, 0, 34],
         [20, 0, 34]]

    events_no_replacement = \
        [[1, 0, 1],
         [1, 0, 12],
         [1, 0, 1234],
         [3, 0, 2],
         [3, 0, 12],
         [3, 0, 1234],
         [10, 0, 3],
         [10, 0, 34],
         [10, 0, 1234],
         [20, 0, 4],
         [20, 0, 34],
         [20, 0, 1234]]

    for replace_events, events_good in [(True, events_replacement),
                                        (False, events_no_replacement)]:
        events = merge_events(events_orig, [1, 2], 12, replace_events)
        events = merge_events(events, [3, 4], 34, replace_events)
        events = merge_events(events, [1, 2, 3, 4], 1234, replace_events)
        assert_array_equal(events, events_good)


def test_io_events():
    """Test IO for events."""
    tempdir = _TempDir()
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        events2 = read_events(fname_txt_mpr, mask=0, mask_type='not_and')
        assert_true(sum('first row of' in str(ww.message) for ww in w) == 1)
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
    c = read_events(op.join(tempdir, 'events-eve.fif'),
                    exclude=[2, 3, 4, 5, 32])
    d = read_events(op.join(tempdir, 'events-eve.fif'), include=1,
                    exclude=[2, 3])
    assert_array_equal(a, b)
    assert_array_equal(a, c)
    assert_array_equal(a, d)

    # test reading file with mask=None
    events2 = events.copy()
    events2[:, -1] = range(events2.shape[0])
    write_events(op.join(tempdir, 'events-eve.fif'), events2)
    events3 = read_events(op.join(tempdir, 'events-eve.fif'), mask=None)
    assert_array_almost_equal(events2, events3)

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
    assert_naming(w, 'test_event.py', 2)


def test_find_events():
    """Test find events in raw file."""
    events = read_events(fname)
    raw = read_raw_fif(raw_fname, preload=True)
    # let's test the defaulting behavior while we're at it
    extra_ends = ['', '_1']
    orig_envs = [os.getenv('MNE_STIM_CHANNEL%s' % s) for s in extra_ends]
    os.environ['MNE_STIM_CHANNEL'] = 'STI 014'
    if 'MNE_STIM_CHANNEL_1' in os.environ:
        del os.environ['MNE_STIM_CHANNEL_1']
    events2 = find_events(raw)
    assert_array_almost_equal(events, events2)
    # now test with mask
    events11 = find_events(raw, mask=3, mask_type='not_and')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        events22 = read_events(fname, mask=3)
        assert_true(sum('events masked' in str(ww.message) for ww in w) == 1)
    assert_array_equal(events11, events22)

    # Reset some data for ease of comparison
    raw._first_samps[0] = 0
    raw.info['sfreq'] = 1000
    raw._update_times()

    stim_channel = 'STI 014'
    stim_channel_idx = pick_channels(raw.info['ch_names'],
                                     include=[stim_channel])

    # test digital masking
    raw._data[stim_channel_idx, :5] = np.arange(5)
    raw._data[stim_channel_idx, 5:] = 0
    # 1 == '0b1', 2 == '0b10', 3 == '0b11', 4 == '0b100'

    assert_raises(TypeError, find_events, raw, mask="0")
    assert_raises(ValueError, find_events, raw, mask=0, mask_type='blah')
    # testing mask_type. default = 'not_and'
    assert_array_equal(find_events(raw, shortest_event=1, mask=1,
                                   mask_type='not_and'),
                       [[2, 0, 2], [4, 2, 4]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=2,
                                   mask_type='not_and'),
                       [[1, 0, 1], [3, 0, 1], [4, 1, 4]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=3,
                                   mask_type='not_and'),
                       [[4, 0, 4]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=4,
                                   mask_type='not_and'),
                       [[1, 0, 1], [2, 1, 2], [3, 2, 3]])
    # testing with mask_type = 'and'
    assert_array_equal(find_events(raw, shortest_event=1, mask=1,
                       mask_type='and'),
                       [[1, 0, 1], [3, 0, 1]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=2,
                       mask_type='and'),
                       [[2, 0, 2]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=3,
                       mask_type='and'),
                       [[1, 0, 1], [2, 1, 2], [3, 2, 3]])
    assert_array_equal(find_events(raw, shortest_event=1, mask=4,
                       mask_type='and'),
                       [[4, 0, 4]])

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
    assert_raises(ValueError, find_events, raw, output='step',
                  consecutive=True)
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


def test_pick_events():
    """Test pick events in a events ndarray."""
    events = np.array([[1, 0, 1],
                       [2, 1, 0],
                       [3, 0, 4],
                       [4, 4, 2],
                       [5, 2, 0]])
    assert_array_equal(pick_events(events, include=[1, 4], exclude=4),
                       [[1, 0, 1],
                        [3, 0, 4]])
    assert_array_equal(pick_events(events, exclude=[0, 2]),
                       [[1, 0, 1],
                        [3, 0, 4]])
    assert_array_equal(pick_events(events, include=[1, 2], step=True),
                       [[1, 0, 1],
                        [2, 1, 0],
                        [4, 4, 2],
                        [5, 2, 0]])


def test_make_fixed_length_events():
    """Test making events of a fixed length."""
    raw = read_raw_fif(raw_fname)
    events = make_fixed_length_events(raw, id=1)
    assert_true(events.shape[1], 3)
    events_zero = make_fixed_length_events(raw, 1, first_samp=False)
    assert_equal(events_zero[0, 0], 0)
    assert_array_equal(events_zero[:, 0], events[:, 0] - raw.first_samp)
    # With limits
    tmin, tmax = raw.times[[0, -1]]
    duration = tmax - tmin
    events = make_fixed_length_events(raw, 1, tmin, tmax, duration)
    assert_equal(events.shape[0], 1)
    # With bad limits (no resulting events)
    assert_raises(ValueError, make_fixed_length_events, raw, 1,
                  tmin, tmax - 1e-3, duration)
    # not raw, bad id or duration
    assert_raises(ValueError, make_fixed_length_events, raw, 2.3)
    assert_raises(ValueError, make_fixed_length_events, 'not raw', 2)
    assert_raises(ValueError, make_fixed_length_events, raw, 23, tmin, tmax,
                  'abc')


def test_define_events():
    """Test defining response events."""
    events = read_events(fname)
    raw = read_raw_fif(raw_fname)
    events_, _ = define_target_events(events, 5, 32, raw.info['sfreq'],
                                      .2, 0.7, 42, 99)
    n_target = events[events[:, 2] == 5].shape[0]
    n_miss = events_[events_[:, 2] == 99].shape[0]
    n_target_ = events_[events_[:, 2] == 42].shape[0]

    assert_true(n_target_ == (n_target - n_miss))

    events = np.array([[0, 0, 1],
                       [375, 0, 2],
                       [500, 0, 1],
                       [875, 0, 3],
                       [1000, 0, 1],
                       [1375, 0, 3],
                       [1100, 0, 1],
                       [1475, 0, 2],
                       [1500, 0, 1],
                       [1875, 0, 2]])
    true_lag_nofill = [1500., 1500., 1500.]
    true_lag_fill = [1500., np.nan, np.nan, 1500., 1500.]
    n, lag_nofill = define_target_events(events, 1, 2, 250., 1.4, 1.6, 5)
    n, lag_fill = define_target_events(events, 1, 2, 250., 1.4, 1.6, 5, 99)

    assert_array_equal(true_lag_fill, lag_fill)
    assert_array_equal(true_lag_nofill, lag_nofill)


@testing.requires_testing_data
def test_acqparser():
    """ Test AcqParserFIF """
    # no acquisition parameters
    assert_raises(ValueError, AcqParserFIF, {'acq_pars': ''})
    # invalid acquisition parameters
    assert_raises(ValueError, AcqParserFIF, {'acq_pars': 'baaa'})
    assert_raises(ValueError, AcqParserFIF, {'acq_pars': 'ERFVersion\n1'})
    # test oldish file
    raw = read_raw_fif(raw_fname, preload=False)
    acqp = AcqParserFIF(raw.info)
    # test __repr__()
    assert_true(repr(acqp))
    # old file should trigger compat mode
    assert_true(acqp.compat)
    # count events and categories
    assert_equal(len(acqp.categories), 6)
    assert_equal(len(acqp._categories), 17)
    assert_equal(len(acqp.events), 6)
    assert_equal(len(acqp._events), 17)
    # get category
    assert_true(acqp['Surprise visual'])
    # test TRIUX file
    raw = read_raw_fif(fname_raw_elekta, preload=False)
    acqp = raw.acqparser
    assert_true(acqp is raw.acqparser)  # same one, not regenerated
    # test __repr__()
    assert_true(repr(acqp))
    # this file should not be in compatibility mode
    assert_true(not acqp.compat)
    # nonexisting category
    assert_raises(KeyError, acqp.__getitem__, 'does not exist')
    assert_raises(KeyError, acqp.get_condition, raw, 'foo')
    # category not a string
    assert_raises(ValueError, acqp.__getitem__, 0)
    # number of events / categories
    assert_equal(len(acqp), 7)
    assert_equal(len(acqp.categories), 7)
    assert_equal(len(acqp._categories), 32)
    assert_equal(len(acqp.events), 6)
    assert_equal(len(acqp._events), 32)
    # get category
    assert_true(acqp['Test event 5'])


@testing.requires_testing_data
def test_acqparser_averaging():
    """ Test averaging with AcqParserFIF vs. Elekta software """
    raw = read_raw_fif(fname_raw_elekta, preload=True)
    acqp = AcqParserFIF(raw.info)
    for cat in acqp.categories:
        # XXX datasets match only when baseline is applied to both,
        # not sure where relative dc shift comes from
        cond = acqp.get_condition(raw, cat)
        eps = Epochs(raw, baseline=(-.05, 0), **cond)
        ev = eps.average()
        ev_ref = read_evokeds(fname_ave_elekta, cat['comment'],
                              baseline=(-.05, 0), proj=False)
        ev_mag = ev.copy()
        ev_mag.pick_channels(['MEG0111'])
        ev_grad = ev.copy()
        ev_grad.pick_channels(['MEG2643', 'MEG1622'])
        ev_ref_mag = ev_ref.copy()
        ev_ref_mag.pick_channels(['MEG0111'])
        ev_ref_grad = ev_ref.copy()
        ev_ref_grad.pick_channels(['MEG2643', 'MEG1622'])
        assert_allclose(ev_mag.data, ev_ref_mag.data,
                        rtol=0, atol=1e-15)  # tol = 1 fT
        assert_allclose(ev_grad.data, ev_ref_grad.data,
                        rtol=0, atol=1e-13)  # tol = 1 fT/cm

run_tests_if_main()
