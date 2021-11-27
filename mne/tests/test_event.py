# -*- coding: utf-8 -*-
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
import os.path as op
import os

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
import pytest

from mne import (read_events, write_events, make_fixed_length_events,
                 find_events, pick_events, find_stim_steps, pick_channels,
                 read_evokeds, Epochs, create_info, compute_raw_covariance,
                 Annotations)
from mne.io import read_raw_fif, RawArray
from mne.event import (define_target_events, merge_events, AcqParserFIF,
                       shift_time_events)
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname = op.join(base_dir, 'test-eve.fif')
fname_raw = op.join(base_dir, 'test_raw.fif')
fname_gz = op.join(base_dir, 'test-eve.fif.gz')
fname_1 = op.join(base_dir, 'test-1-eve.fif')
fname_txt = op.join(base_dir, 'test-eve.eve')
fname_txt_1 = op.join(base_dir, 'test-eve-1.eve')
fname_c_annot = op.join(base_dir, 'test_raw-annot.fif')

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
    with pytest.warns(RuntimeWarning, match='STI016'):
        events = find_events(raw, 'STI 014')
    assert_array_equal(events[0], [raw.first_samp + 1, 0, 32765])
    events = find_events(raw, 'STI 014', uint_cast=True)
    assert_array_equal(events[0], [raw.first_samp + 1, 0, 32771])


def test_add_events():
    """Test adding events to a Raw file."""
    # need preload
    raw = read_raw_fif(raw_fname)
    events = np.array([[raw.first_samp, 0, 1]])
    pytest.raises(RuntimeError, raw.add_events, events, 'STI 014')
    raw = read_raw_fif(raw_fname, preload=True)
    orig_events = find_events(raw, 'STI 014')
    # add some events
    events = np.array([raw.first_samp, 0, 1])
    pytest.raises(ValueError, raw.add_events, events, 'STI 014')  # bad shape
    events[0] = raw.first_samp + raw.n_times + 1
    events = events[np.newaxis, :]
    pytest.raises(ValueError, raw.add_events, events, 'STI 014')  # bad time
    events[0, 0] = raw.first_samp - 1
    pytest.raises(ValueError, raw.add_events, events, 'STI 014')  # bad time
    events[0, 0] = raw.first_samp + 1  # can't actually be first_samp
    pytest.raises(ValueError, raw.add_events, events, 'STI FOO')
    raw.add_events(events, 'STI 014')
    new_events = find_events(raw, 'STI 014')
    assert_array_equal(new_events, np.concatenate((events, orig_events)))
    raw.add_events(events, 'STI 014', replace=True)
    new_events = find_events(raw, 'STI 014')
    assert_array_equal(new_events, events)


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


def test_io_events(tmp_path):
    """Test IO for events."""
    # Test binary fif IO
    events = read_events(fname)  # Use as the gold standard
    fname_temp = tmp_path / 'events-eve.fif'
    write_events(fname_temp, events)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)

    # Test binary fif.gz IO
    events2 = read_events(fname_gz)  # Use as the gold standard
    assert_array_almost_equal(events, events2)
    fname_temp = str(fname_temp) + '.gz'
    write_events(fname_temp, events2)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)

    # Test new format text file IO
    fname_temp = tmp_path / 'events.eve'
    write_events(fname_temp, events)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)
    with pytest.warns(RuntimeWarning, match='first row of'):
        events2 = read_events(fname_txt_mpr, mask=0, mask_type='not_and')
    assert_array_almost_equal(events, events2)

    # Test old format text file IO
    events2 = read_events(fname_old_txt)
    assert_array_almost_equal(events, events2)
    write_events(fname_temp, events)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)

    # Test event selection
    fname_temp = tmp_path / 'events-eve.fif'
    a = read_events(fname_temp, include=1)
    b = read_events(fname_temp, include=[1])
    c = read_events(fname_temp, exclude=[2, 3, 4, 5, 32])
    d = read_events(fname_temp, include=1, exclude=[2, 3])
    assert_array_equal(a, b)
    assert_array_equal(a, c)
    assert_array_equal(a, d)

    # test reading file with mask=None
    events2 = events.copy()
    events2[:, -1] = range(events2.shape[0])
    write_events(fname_temp, events2)
    events3 = read_events(fname_temp, mask=None)
    assert_array_almost_equal(events2, events3)

    # Test binary file IO for 1 event
    events = read_events(fname_1)  # Use as the new gold standard
    write_events(fname_temp, events)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)

    # Test text file IO for 1 event
    fname_temp = tmp_path / 'events.eve'
    write_events(fname_temp, events)
    events2 = read_events(fname_temp)
    assert_array_almost_equal(events, events2)

    # test warnings on bad filenames
    fname2 = tmp_path / 'test-bad-name.fif'
    with pytest.warns(RuntimeWarning, match='-eve.fif'):
        write_events(fname2, events)
    with pytest.warns(RuntimeWarning, match='-eve.fif'):
        read_events(fname2)

    # No event_id
    with pytest.raises(RuntimeError, match='No event_id'):
        read_events(fname, return_event_id=True)


def test_io_c_annot():
    """Test I/O of MNE-C -annot.fif files."""
    raw = read_raw_fif(fname_raw)
    sfreq, first_samp = raw.info['sfreq'], raw.first_samp
    events = read_events(fname_c_annot)
    events_2, event_id = read_events(fname_c_annot, return_event_id=True)
    assert_array_equal(events_2, events)
    expected = np.arange(2, 5) * sfreq + first_samp
    assert_allclose(events[:, 0], expected, atol=3)  # clicking accuracy (samp)
    expected = {'Two sec': 1001, 'Three and four sec': 1002}
    assert event_id == expected


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
    with pytest.warns(RuntimeWarning, match='events masked'):
        events22 = read_events(fname, mask=3, mask_type='not_and')
    assert_array_equal(events11, events22)

    # Reset some data for ease of comparison
    raw._first_samps[0] = 0
    with raw.info._unlock():
        raw.info['sfreq'] = 1000

    stim_channel = 'STI 014'
    stim_channel_idx = pick_channels(raw.info['ch_names'],
                                     include=[stim_channel])

    # test digital masking
    raw._data[stim_channel_idx, :5] = np.arange(5)
    raw._data[stim_channel_idx, 5:] = 0
    # 1 == '0b1', 2 == '0b10', 3 == '0b11', 4 == '0b100'

    pytest.raises(TypeError, find_events, raw, mask="0", mask_type='and')
    pytest.raises(ValueError, find_events, raw, mask=0, mask_type='blah')
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
    pytest.raises(ValueError, find_events, raw, output='step',
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

    # Test with list of stim channels
    raw._data[stim_channel_idx, 1:101] = np.zeros(100)
    raw._data[stim_channel_idx, 10:11] = 1
    raw._data[stim_channel_idx, 30:31] = 3
    stim_channel2 = 'STI 015'
    stim_channel2_idx = pick_channels(raw.info['ch_names'],
                                      include=[stim_channel2])
    raw._data[stim_channel2_idx, :] = 0
    raw._data[stim_channel2_idx, :100] = raw._data[stim_channel_idx, 5:105]
    events1 = find_events(raw, stim_channel='STI 014')
    events2 = events1.copy()
    events2[:, 0] -= 5
    events = find_events(raw, stim_channel=['STI 014', stim_channel2])
    assert_array_equal(events[::2], events2)
    assert_array_equal(events[1::2], events1)

    # test initial_event argument
    info = create_info(['MYSTI'], 1000, 'stim')
    data = np.zeros((1, 1000))
    raw = RawArray(data, info)
    data[0, :10] = 100
    data[0, 30:40] = 200
    assert_array_equal(find_events(raw, 'MYSTI'), [[30, 0, 200]])
    assert_array_equal(find_events(raw, 'MYSTI', initial_event=True),
                       [[0, 0, 100], [30, 0, 200]])

    # test error message for raw without stim channels
    raw = read_raw_fif(raw_fname, preload=True)
    raw.pick_types(meg=True, stim=False)
    # raw does not have annotations
    with pytest.raises(ValueError, match="'stim_channel'"):
        find_events(raw)
    # if raw has annotations, we show a different error message
    raw.set_annotations(Annotations(0, 2, "test"))
    with pytest.raises(ValueError, match="mne.events_from_annotations"):
        find_events(raw)


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
    assert events.shape[1] == 3
    events_zero = make_fixed_length_events(raw, 1, first_samp=False)
    assert_equal(events_zero[0, 0], 0)
    assert_array_equal(events_zero[:, 0], events[:, 0] - raw.first_samp)
    # With limits
    tmin, tmax = raw.times[[0, -1]]
    duration = tmax - tmin
    events = make_fixed_length_events(raw, 1, tmin, tmax, duration)
    assert_equal(events.shape[0], 1)
    # With bad limits (no resulting events)
    pytest.raises(ValueError, make_fixed_length_events, raw, 1,
                  tmin, tmax - 1e-3, duration)
    # not raw, bad id or duration
    pytest.raises(TypeError, make_fixed_length_events, raw, 2.3)
    pytest.raises(TypeError, make_fixed_length_events, 'not raw', 2)
    pytest.raises(TypeError, make_fixed_length_events, raw, 23, tmin, tmax,
                  'abc')

    # Let's try some ugly sample rate/sample count combos
    data = np.random.RandomState(0).randn(1, 27768)

    # This breaks unless np.round() is used in make_fixed_length_events
    info = create_info(1, 155.4499969482422)
    raw = RawArray(data, info)
    events = make_fixed_length_events(raw, 1, duration=raw.times[-1])
    assert events[0, 0] == 0
    assert len(events) == 1

    # Without use_rounding=True this breaks
    raw = RawArray(data[:, :21216], info)
    events = make_fixed_length_events(raw, 1, duration=raw.times[-1])
    assert events[0, 0] == 0
    assert len(events) == 1

    # Make sure it gets used properly by compute_raw_covariance
    cov = compute_raw_covariance(raw, tstep=None)
    expected = np.cov(data[:, :21216])
    assert_allclose(cov['data'], expected, atol=1e-12)

    # overlaps
    events = make_fixed_length_events(raw, 1, duration=1)
    assert len(events) == 136
    events_ol = make_fixed_length_events(raw, 1, duration=1, overlap=0.5)
    assert len(events_ol) == 271
    events_ol_2 = make_fixed_length_events(raw, 1, duration=1, overlap=0.9)
    assert len(events_ol_2) == 1355
    assert_array_equal(events_ol_2[:, 0], np.unique(events_ol_2[:, 0]))
    with pytest.raises(ValueError, match='overlap must be'):
        make_fixed_length_events(raw, 1, duration=1, overlap=1.1)


def test_define_events():
    """Test defining response events."""
    events = read_events(fname)
    raw = read_raw_fif(raw_fname)
    events_, _ = define_target_events(events, 5, 32, raw.info['sfreq'],
                                      .2, 0.7, 42, 99)
    n_target = events[events[:, 2] == 5].shape[0]
    n_miss = events_[events_[:, 2] == 99].shape[0]
    n_target_ = events_[events_[:, 2] == 42].shape[0]

    assert (n_target_ == (n_target - n_miss))

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
    """Test AcqParserFIF."""
    # no acquisition parameters
    pytest.raises(ValueError, AcqParserFIF, {'acq_pars': ''})
    # invalid acquisition parameters
    pytest.raises(ValueError, AcqParserFIF, {'acq_pars': 'baaa'})
    pytest.raises(ValueError, AcqParserFIF, {'acq_pars': 'ERFVersion\n1'})
    # test oldish file
    raw = read_raw_fif(raw_fname, preload=False)
    acqp = AcqParserFIF(raw.info)
    # test __repr__()
    assert (repr(acqp))
    # old file should trigger compat mode
    assert (acqp.compat)
    # count events and categories
    assert_equal(len(acqp.categories), 6)
    assert_equal(len(acqp._categories), 17)
    assert_equal(len(acqp.events), 6)
    assert_equal(len(acqp._events), 17)
    # get category
    assert (acqp['Surprise visual'])
    # test TRIUX file
    raw = read_raw_fif(fname_raw_elekta, preload=False)
    acqp = raw.acqparser
    assert (acqp is raw.acqparser)  # same one, not regenerated
    # test __repr__()
    assert (repr(acqp))
    # this file should not be in compatibility mode
    assert (not acqp.compat)
    # nonexistent category
    pytest.raises(KeyError, acqp.__getitem__, 'does not exist')
    pytest.raises(KeyError, acqp.get_condition, raw, 'foo')
    # category not a string
    pytest.raises(TypeError, acqp.__getitem__, 0)
    # number of events / categories
    assert_equal(len(acqp), 7)
    assert_equal(len(acqp.categories), 7)
    assert_equal(len(acqp._categories), 32)
    assert_equal(len(acqp.events), 6)
    assert_equal(len(acqp._events), 32)
    # get category
    assert (acqp['Test event 5'])


@testing.requires_testing_data
def test_acqparser_averaging():
    """Test averaging with AcqParserFIF vs. Elekta software."""
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
        # Elekta put these in a different order
        assert ev_grad.ch_names[::-1] == ev_ref_grad.ch_names
        assert_allclose(ev_grad.data[::-1], ev_ref_grad.data,
                        rtol=0, atol=1e-13)  # tol = 1 fT/cm


def test_shift_time_events():
    """Test events latency shift by a given amount."""
    events = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    EXPECTED = [1, 2, 3]
    new_events = shift_time_events(events, ids=None, tshift=1, sfreq=1)
    assert all(new_events[:, 0] == EXPECTED)

    events = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    EXPECTED = [0, 2, 3]
    new_events = shift_time_events(events, ids=[1, 2], tshift=1, sfreq=1)
    assert all(new_events[:, 0] == EXPECTED)
