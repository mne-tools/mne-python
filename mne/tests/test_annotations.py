# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

from datetime import datetime, timezone
from itertools import repeat
from collections import OrderedDict
import sys

import os.path as op

import pytest
from pytest import approx
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose)

import numpy as np

import mne
from mne import (create_info, read_annotations, annotations_from_events,
                 events_from_annotations)
from mne import Epochs, Annotations
from mne.utils import (run_tests_if_main, _TempDir, requires_version,
                       catch_logging)
from mne.utils import (assert_and_remove_boundary_annot, _raw_annot,
                       _dt_to_stamp, _stamp_to_dt)
from mne.io import read_raw_fif, RawArray, concatenate_raws
from mne.annotations import (_sync_onset, _handle_meas_date,
                             _read_annotations_txt_parse_header)
from mne.datasets import testing

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data',
                    'test_raw.fif')

first_samps = pytest.mark.parametrize('first_samp', (0, 10000))


# On Windows, datetime.fromtimestamp throws an error for negative times.
# We mimic this behavior on non-Windows platforms for ease of testing.
class _windows_datetime(datetime):
    @classmethod
    def fromtimestamp(cls, timestamp, tzinfo=None):
        if timestamp < 0:
            raise OSError('[Errno 22] Invalid argument')
        return datetime.fromtimestamp(timestamp, tzinfo)


@pytest.fixture(scope='function')
def windows_like_datetime(monkeypatch):
    """Ensure datetime.fromtimestamp is Windows-like."""
    if not sys.platform.startswith('win'):
        monkeypatch.setattr('mne.annotations.datetime', _windows_datetime)
    yield


def test_basics():
    """Test annotation class."""
    raw = read_raw_fif(fif_fname)
    assert raw.annotations is not None  # XXX to be fixed in #5416
    assert len(raw.annotations.onset) == 0  # XXX to be fixed in #5416
    pytest.raises(IOError, read_annotations, fif_fname)
    onset = np.array(range(10))
    duration = np.ones(10)
    description = np.repeat('test', 10)
    dt = raw.info['meas_date']
    assert isinstance(dt, datetime)
    stamp = _dt_to_stamp(dt)
    # Test time shifts.
    for orig_time in [None, dt, stamp[0], stamp]:
        annot = Annotations(onset, duration, description, orig_time)
        if orig_time is None:
            assert annot.orig_time is None
        else:
            assert isinstance(annot.orig_time, datetime)
            assert annot.orig_time.tzinfo is timezone.utc

    pytest.raises(ValueError, Annotations, onset, duration, description[:9])
    pytest.raises(ValueError, Annotations, [onset, 1], duration, description)
    pytest.raises(ValueError, Annotations, onset, [duration, 1], description)

    # Test combining annotations with concatenate_raws
    raw2 = raw.copy()
    delta = raw.times[-1] + 1. / raw.info['sfreq']
    orig_time = (stamp[0] + stamp[1] * 1e-6 + raw2._first_time)
    offset = _dt_to_stamp(_handle_meas_date(raw2.info['meas_date']))
    offset = offset[0] + offset[1] * 1e-6
    offset = orig_time - offset
    assert_allclose(offset, raw._first_time)
    annot = Annotations(onset, duration, description, orig_time)
    assert annot.orig_time is not None
    assert ' segments' in repr(annot)
    raw2.set_annotations(annot)
    assert_allclose(raw2.annotations.onset, onset + offset)
    assert raw2.annotations is not annot
    assert raw2.annotations.orig_time is not None
    concatenate_raws([raw, raw2])
    assert_and_remove_boundary_annot(raw)
    assert_allclose(onset + offset + delta, raw.annotations.onset, rtol=1e-5)
    assert_array_equal(annot.duration, raw.annotations.duration)
    assert_array_equal(raw.annotations.description, np.repeat('test', 10))


def test_raw_array_orig_times():
    """Test combining with RawArray and orig_times."""
    data = np.random.randn(2, 1000) * 10e-12
    sfreq = 100.
    info = create_info(ch_names=['MEG1', 'MEG2'], ch_types=['grad'] * 2,
                       sfreq=sfreq)
    meas_date = _handle_meas_date(np.pi)
    info['meas_date'] = meas_date
    raws = []
    for first_samp in [12300, 100, 12]:
        raw = RawArray(data.copy(), info, first_samp=first_samp)
        ants = Annotations([1., 2.], [.5, .5], 'x', np.pi + first_samp / sfreq)
        raw.set_annotations(ants)
        raws.append(raw)
    assert_allclose(raws[0].annotations.onset, [124, 125])
    raw = RawArray(data.copy(), info)
    assert not len(raw.annotations)
    raw.set_annotations(Annotations([1.], [.5], 'x', None))
    assert_allclose(raw.annotations.onset, [1.])
    raws.append(raw)
    raw = concatenate_raws(raws, verbose='debug')
    assert raw.info['meas_date'] == raw.annotations.orig_time == meas_date
    assert_and_remove_boundary_annot(raw, 3)
    assert_array_equal(raw.annotations.onset, [124., 125., 134., 135.,
                                               144., 145., 154.])
    raw.annotations.delete(2)
    assert_array_equal(raw.annotations.onset, [124., 125., 135., 144.,
                                               145., 154.])
    raw.annotations.append(5, 1.5, 'y')
    assert_array_equal(raw.annotations.onset,
                       [5., 124., 125., 135., 144., 145., 154.])
    assert_array_equal(raw.annotations.duration,
                       [1.5, .5, .5, .5, .5, .5, .5])
    assert_array_equal(raw.annotations.description,
                       ['y', 'x', 'x', 'x', 'x', 'x', 'x'])

    # These three things should be equivalent
    stamp = _dt_to_stamp(raw.info['meas_date'])
    orig_time = _handle_meas_date(stamp)
    for empty_annot in (
            Annotations([], [], [], stamp),
            Annotations([], [], [], orig_time),
            Annotations([], [], [], None),
            None):
        raw.set_annotations(empty_annot)
        assert isinstance(raw.annotations, Annotations)
        assert len(raw.annotations) == 0
        assert raw.annotations.orig_time == orig_time


def test_crop():
    """Test cropping with annotations."""
    raw = read_raw_fif(fif_fname)
    events = mne.find_events(raw)
    onset = events[events[:, 2] == 1, 0] / raw.info['sfreq']
    duration = np.full_like(onset, 0.5)
    description = ['bad %d' % k for k in range(len(onset))]
    annot = mne.Annotations(onset, duration, description,
                            orig_time=raw.info['meas_date'])
    raw.set_annotations(annot)

    split_time = raw.times[-1] / 2. + 2.
    split_idx = len(onset) // 2 + 1
    raw_cropped_left = raw.copy().crop(0., split_time - 1. / raw.info['sfreq'])
    assert_array_equal(raw_cropped_left.annotations.description,
                       raw.annotations.description[:split_idx])
    assert_allclose(raw_cropped_left.annotations.duration,
                    raw.annotations.duration[:split_idx])
    assert_allclose(raw_cropped_left.annotations.onset,
                    raw.annotations.onset[:split_idx])
    raw_cropped_right = raw.copy().crop(split_time, None)
    assert_array_equal(raw_cropped_right.annotations.description,
                       raw.annotations.description[split_idx:])
    assert_allclose(raw_cropped_right.annotations.duration,
                    raw.annotations.duration[split_idx:])
    assert_allclose(raw_cropped_right.annotations.onset,
                    raw.annotations.onset[split_idx:])
    raw_concat = mne.concatenate_raws([raw_cropped_left, raw_cropped_right],
                                      verbose='debug')
    assert_allclose(raw_concat.times, raw.times)
    assert_allclose(raw_concat[:][0], raw[:][0], atol=1e-20)
    assert_and_remove_boundary_annot(raw_concat)
    # Ensure we annotations survive round-trip crop->concat
    assert_array_equal(raw_concat.annotations.description,
                       raw.annotations.description)
    for attr in ('onset', 'duration'):
        assert_allclose(getattr(raw_concat.annotations, attr),
                        getattr(raw.annotations, attr),
                        err_msg='Failed for %s:' % (attr,))

    raw.set_annotations(None)  # undo

    # Test concatenating annotations with and without orig_time.
    raw2 = raw.copy()
    raw.set_annotations(Annotations([45.], [3], 'test', raw.info['meas_date']))
    raw2.set_annotations(Annotations([2.], [3], 'BAD', None))
    expected_onset = [45., 2. + raw._last_time]
    raw = concatenate_raws([raw, raw2])
    assert_and_remove_boundary_annot(raw)
    assert_array_almost_equal(raw.annotations.onset, expected_onset, decimal=2)

    # Test IO
    tempdir = _TempDir()
    fname = op.join(tempdir, 'test-annot.fif')
    raw.annotations.save(fname)
    annot_read = read_annotations(fname)
    for attr in ('onset', 'duration'):
        assert_allclose(getattr(annot_read, attr),
                        getattr(raw.annotations, attr))
    assert annot_read.orig_time == raw.annotations.orig_time
    assert_array_equal(annot_read.description, raw.annotations.description)
    annot = Annotations((), (), ())
    annot.save(fname)
    pytest.raises(IOError, read_annotations, fif_fname)  # none in old raw
    annot = read_annotations(fname)
    assert isinstance(annot, Annotations)
    assert len(annot) == 0
    annot.crop()  # test if cropping empty annotations doesn't raise an error
    # Test that empty annotations can be saved with an object
    fname = op.join(tempdir, 'test_raw.fif')
    raw.set_annotations(annot)
    raw.save(fname)
    raw_read = read_raw_fif(fname)
    assert isinstance(raw_read.annotations, Annotations)
    assert len(raw_read.annotations) == 0
    raw.set_annotations(None)
    raw.save(fname, overwrite=True)
    raw_read = read_raw_fif(fname)
    assert raw_read.annotations is not None  # XXX to be fixed in #5416
    assert len(raw_read.annotations.onset) == 0  # XXX to be fixed in #5416


@first_samps
def test_chunk_duration(first_samp):
    """Test chunk_duration."""
    # create dummy raw
    raw = RawArray(data=np.empty([10, 10], dtype=np.float64),
                   info=create_info(ch_names=10, sfreq=1.),
                   first_samp=first_samp)
    raw.info['meas_date'] = _handle_meas_date(0)
    raw.set_annotations(Annotations(description='foo', onset=[0],
                                    duration=[10], orig_time=None))
    assert raw.annotations.orig_time == raw.info['meas_date']
    assert_allclose(raw.annotations.onset, [first_samp])

    # expected_events = [[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1], ..
    #                    [9, 0, 1], [9, 0, 1]]
    expected_events = np.atleast_2d(np.repeat(range(10), repeats=2)).T
    expected_events = np.insert(expected_events, 1, 0, axis=1)
    expected_events = np.insert(expected_events, 2, 1, axis=1)
    expected_events[:, 0] += first_samp

    events, events_id = events_from_annotations(raw, chunk_duration=.5,
                                                use_rounding=False)
    assert_array_equal(events, expected_events)

    # test chunk durations that do not fit equally in annotation duration
    expected_events = np.zeros((3, 3))
    expected_events[:, -1] = 1
    expected_events[:, 0] = np.arange(0, 9, step=3) + first_samp
    events, events_id = events_from_annotations(raw, chunk_duration=3.)
    assert_array_equal(events, expected_events)


def test_crop_more():
    """Test more cropping."""
    raw = mne.io.read_raw_fif(fif_fname).crop(0, 11).load_data()
    raw._data[:] = np.random.RandomState(0).randn(*raw._data.shape)
    onset = np.array([0.47058824, 2.49773765, 6.67873287, 9.15837097])
    duration = np.array([0.89592767, 1.13574672, 1.09954739, 0.48868752])
    annotations = mne.Annotations(onset, duration, 'BAD')
    raw.set_annotations(annotations)
    assert len(raw.annotations) == 4
    delta = 1. / raw.info['sfreq']
    offset = raw.first_samp * delta
    raw_concat = mne.concatenate_raws(
        [raw.copy().crop(0, 4 - delta),
         raw.copy().crop(4, 8 - delta),
         raw.copy().crop(8, None)])
    assert_allclose(raw_concat.times, raw.times)
    assert_allclose(raw_concat[:][0], raw[:][0])
    assert raw_concat.first_samp == raw.first_samp
    assert_and_remove_boundary_annot(raw_concat, 2)
    assert len(raw_concat.annotations) == 4
    assert_array_equal(raw_concat.annotations.description,
                       raw.annotations.description)
    assert_allclose(raw.annotations.duration, duration)
    assert_allclose(raw_concat.annotations.duration, duration)
    assert_allclose(raw.annotations.onset, onset + offset)
    assert_allclose(raw_concat.annotations.onset, onset + offset,
                    atol=1. / raw.info['sfreq'])


@testing.requires_testing_data
def test_read_brainstorm_annotations():
    """Test reading for Brainstorm events file."""
    fname = op.join(data_dir, 'events_sample_audvis_raw_bst.mat')
    annot = read_annotations(fname)
    assert len(annot) == 238
    assert annot.onset.min() > 40  # takes into account first_samp
    assert np.unique(annot.description).size == 5


@first_samps
def test_raw_reject(first_samp):
    """Test raw data getter with annotation reject."""
    sfreq = 100.
    info = create_info(['a', 'b', 'c', 'd', 'e'], sfreq, ch_types='eeg')
    raw = RawArray(np.ones((5, 15000)), info, first_samp=first_samp)
    with pytest.warns(RuntimeWarning, match='outside the data range'):
        raw.set_annotations(Annotations([2, 100, 105, 148],
                                        [2, 8, 5, 8], 'BAD'))
    data, times = raw.get_data([0, 1, 3, 4], 100, 11200,  # 1-112 sec
                               'omit', return_times=True)
    bad_times = np.concatenate([np.arange(200, 400),
                                np.arange(10000, 10800),
                                np.arange(10500, 11000)])
    expected_times = np.setdiff1d(np.arange(100, 11200), bad_times) / sfreq
    assert_allclose(times, expected_times)

    # with orig_time and complete overlap
    raw = read_raw_fif(fif_fname)
    raw.set_annotations(Annotations(onset=[1, 4, 5] + raw._first_time,
                                    duration=[1, 3, 1],
                                    description='BAD',
                                    orig_time=raw.info['meas_date']))
    t_stop = 18.
    assert raw.times[-1] > t_stop
    n_stop = int(round(t_stop * raw.info['sfreq']))
    n_drop = int(round(4 * raw.info['sfreq']))
    assert len(raw.times) >= n_stop
    data, times = raw.get_data(range(10), 0, n_stop, 'omit', True)
    assert data.shape == (10, n_stop - n_drop)
    assert times[-1] == raw.times[n_stop - 1]
    assert_array_equal(data[:, -100:], raw[:10, n_stop - 100:n_stop][0])

    data, times = raw.get_data(range(10), 0, n_stop, 'NaN', True)
    assert_array_equal(data.shape, (10, n_stop))
    assert times[-1] == raw.times[n_stop - 1]
    t_1, t_2 = raw.time_as_index([1, 2], use_rounding=True)
    assert np.isnan(data[:, t_1:t_2]).all()  # 1s -2s
    assert not np.isnan(data[:, :t_1].any())
    assert not np.isnan(data[:, t_2:].any())
    assert_array_equal(data[:, -100:], raw[:10, n_stop - 100:n_stop][0])
    assert_array_equal(raw.get_data(), raw[:][0])

    # Test _sync_onset
    times = [10, -88, 190]
    onsets = _sync_onset(raw, times)
    assert_array_almost_equal(onsets, times - raw.first_samp /
                              raw.info['sfreq'])
    assert_array_almost_equal(times, _sync_onset(raw, onsets, True))


@first_samps
def test_annotation_filtering(first_samp):
    """Test that annotations work properly with filtering."""
    # Create data with just a DC component
    data = np.ones((1, 1000))
    info = create_info(1, 1000., 'eeg')
    raws = [RawArray(data * (ii + 1), info, first_samp=first_samp)
            for ii in range(4)]
    kwargs_pass = dict(l_freq=None, h_freq=50., fir_design='firwin')
    kwargs_stop = dict(l_freq=50., h_freq=None, fir_design='firwin')
    # lowpass filter, which should not modify the data
    raws_pass = [raw.copy().filter(**kwargs_pass) for raw in raws]
    # highpass filter, which should zero it out
    raws_stop = [raw.copy().filter(**kwargs_stop) for raw in raws]
    # concat the original and the filtered segments
    raws_concat = concatenate_raws([raw.copy() for raw in raws])
    raws_zero = raws_concat.copy().apply_function(lambda x: x * 0)
    raws_pass_concat = concatenate_raws(raws_pass)
    raws_stop_concat = concatenate_raws(raws_stop)
    # make sure we did something reasonable with our individual-file filtering
    assert_allclose(raws_concat[0][0], raws_pass_concat[0][0], atol=1e-14)
    assert_allclose(raws_zero[0][0], raws_stop_concat[0][0], atol=1e-14)
    # ensure that our Annotations cut up the filtering properly
    raws_concat_pass = raws_concat.copy().filter(skip_by_annotation='edge',
                                                 **kwargs_pass)
    assert_allclose(raws_concat[0][0], raws_concat_pass[0][0], atol=1e-14)
    raws_concat_stop = raws_concat.copy().filter(skip_by_annotation='edge',
                                                 **kwargs_stop)
    assert_allclose(raws_zero[0][0], raws_concat_stop[0][0], atol=1e-14)
    # one last test: let's cut out a section entirely:
    # here the 1-3 second window should be skipped
    raw = raws_concat.copy()
    raw.annotations.append(1. + raw._first_time, 2., 'foo')
    with catch_logging() as log:
        raw.filter(l_freq=50., h_freq=None, fir_design='firwin',
                   skip_by_annotation='foo', verbose='info')
    log = log.getvalue()
    assert '2 contiguous segments' in log
    raw.annotations.append(2. + raw._first_time, 1., 'foo')  # shouldn't change
    with catch_logging() as log:
        raw.filter(l_freq=50., h_freq=None, fir_design='firwin',
                   skip_by_annotation='foo', verbose='info')
    log = log.getvalue()
    assert '2 contiguous segments' in log
    # our filter will zero out anything not skipped:
    mask = np.concatenate((np.zeros(1000), np.ones(2000), np.zeros(1000)))
    expected_data = raws_concat[0][0][0] * mask
    assert_allclose(raw[0][0][0], expected_data, atol=1e-14)

    # Let's try another one
    raw = raws[0].copy()
    raw.set_annotations(Annotations([0.], [0.5], ['BAD_ACQ_SKIP']))
    my_data, times = raw.get_data(reject_by_annotation='omit',
                                  return_times=True)
    assert_allclose(times, raw.times[500:])
    assert my_data.shape == (1, 500)
    raw_filt = raw.copy().filter(skip_by_annotation='bad_acq_skip',
                                 **kwargs_stop)
    expected = data.copy()
    expected[:, 500:] = 0
    assert_allclose(raw_filt[:][0], expected, atol=1e-14)

    raw = raws[0].copy()
    raw.set_annotations(Annotations([0.5], [0.5], ['BAD_ACQ_SKIP']))
    my_data, times = raw.get_data(reject_by_annotation='omit',
                                  return_times=True)
    assert_allclose(times, raw.times[:500])
    assert my_data.shape == (1, 500)
    raw_filt = raw.copy().filter(skip_by_annotation='bad_acq_skip',
                                 **kwargs_stop)
    expected = data.copy()
    expected[:, :500] = 0
    assert_allclose(raw_filt[:][0], expected, atol=1e-14)


@first_samps
def test_annotation_omit(first_samp):
    """Test raw.get_data with annotations."""
    data = np.concatenate([np.ones((1, 1000)), 2 * np.ones((1, 1000))], -1)
    info = create_info(1, 1000., 'eeg')
    raw = RawArray(data, info, first_samp=first_samp)
    raw.set_annotations(Annotations([0.5], [1], ['bad']))
    expected = raw[0][0]
    assert_allclose(raw.get_data(reject_by_annotation=None), expected)
    # nan
    expected[0, 500:1500] = np.nan
    assert_allclose(raw.get_data(reject_by_annotation='nan'), expected)
    got = np.concatenate([raw.get_data(start=start, stop=stop,
                                       reject_by_annotation='nan')
                          for start, stop in ((0, 1000), (1000, 2000))], -1)
    assert_allclose(got, expected)
    # omit
    expected = expected[:, np.isfinite(expected[0])]
    assert_allclose(raw.get_data(reject_by_annotation='omit'), expected)
    got = np.concatenate([raw.get_data(start=start, stop=stop,
                                       reject_by_annotation='omit')
                          for start, stop in ((0, 1000), (1000, 2000))], -1)
    assert_allclose(got, expected)
    pytest.raises(ValueError, raw.get_data, reject_by_annotation='foo')


def test_annotation_epoching():
    """Test that annotations work properly with concatenated edges."""
    # Create data with just a DC component
    data = np.ones((1, 1000))
    info = create_info(1, 1000., 'eeg')
    raw = concatenate_raws([RawArray(data, info) for ii in range(3)])
    assert raw.annotations is not None
    assert len(raw.annotations) == 4
    assert np.in1d(raw.annotations.description, ['BAD boundary']).sum() == 2
    assert np.in1d(raw.annotations.description, ['EDGE boundary']).sum() == 2
    assert_array_equal(raw.annotations.duration, 0.)
    events = np.array([[a, 0, 1] for a in [0, 500, 1000, 1500, 2000]])
    epochs = Epochs(raw, events, tmin=0, tmax=0.999, baseline=None,
                    preload=True)  # 1000 samples long
    assert_equal(len(epochs.drop_log), len(events))
    assert_equal(len(epochs), 3)
    assert_equal([0, 2, 4], epochs.selection)


def test_annotation_concat():
    """Test if two Annotations objects can be concatenated."""
    a = Annotations([1, 2, 3], [5, 5, 8], ["a", "b", "c"])
    b = Annotations([11, 12, 13], [1, 2, 2], ["x", "y", "z"])

    # test + operator (does not modify a or b)
    c = a + b
    assert_array_equal(c.onset, [1, 2, 3, 11, 12, 13])
    assert_array_equal(c.duration, [5, 5, 8, 1, 2, 2])
    assert_array_equal(c.description, ["a", "b", "c", "x", "y", "z"])
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(len(c), 6)

    # test += operator (modifies a in place)
    a += b
    assert_array_equal(a.onset, [1, 2, 3, 11, 12, 13])
    assert_array_equal(a.duration, [5, 5, 8, 1, 2, 2])
    assert_array_equal(a.description, ["a", "b", "c", "x", "y", "z"])
    assert_equal(len(a), 6)
    assert_equal(len(b), 3)

    # test += operator (modifies a in place)
    b._orig_time = _handle_meas_date(1038942070.7201)
    with pytest.raises(ValueError, match='orig_time should be the same'):
        a += b


def test_annotations_crop():
    """Test basic functionality of annotation crop."""
    onset = np.arange(1, 10)
    duration = np.full_like(onset, 10)
    description = ["yy"] * onset.shape[0]

    a = Annotations(onset=onset,
                    duration=duration,
                    description=description,
                    orig_time=0)

    # cropping window larger than annotations --> do not modify
    a_ = a.copy().crop(tmin=-10, tmax=42)
    assert_array_equal(a_.onset, a.onset)
    assert_array_equal(a_.duration, a.duration)

    # cropping with left shifted window
    with pytest.warns(None) as w:
        a_ = a.copy().crop(tmin=0, tmax=4.2)
    assert_array_equal(a_.onset, [1., 2., 3., 4.])
    assert_allclose(a_.duration, [3.2, 2.2, 1.2, 0.2])
    assert len(w) == 0

    # cropping with right shifted window
    with pytest.warns(None) as w:
        a_ = a.copy().crop(tmin=17.8, tmax=22)
    assert_array_equal(a_.onset, [17.8, 17.8])
    assert_allclose(a_.duration, [0.2, 1.2])
    assert len(w) == 0

    # cropping with centered small window
    a_ = a.copy().crop(tmin=11, tmax=12)
    assert_array_equal(a_.onset, [11, 11, 11, 11, 11, 11, 11, 11, 11])
    assert_array_equal(a_.duration, [0, 1, 1, 1, 1, 1, 1, 1, 1])

    # cropping with out-of-bounds window
    with pytest.warns(None) as w:
        a_ = a.copy().crop(tmin=42, tmax=100)
    assert_array_equal(a_.onset, [])
    assert_array_equal(a_.duration, [])
    assert len(w) == 0

    # test error raising
    with pytest.raises(ValueError, match='tmax should be greater than.*tmin'):
        a.copy().crop(tmin=42, tmax=0)

    # test warnings
    with pytest.warns(RuntimeWarning, match='Omitted .* were outside'):
        a.copy().crop(tmin=42, tmax=100, emit_warning=True)
    with pytest.warns(RuntimeWarning, match='Limited .* expanding outside'):
        a.copy().crop(tmin=0, tmax=12, emit_warning=True)


@testing.requires_testing_data
def test_events_from_annot_in_raw_objects():
    """Test basic functionality of events_fron_annot for raw objects."""
    raw = read_raw_fif(fif_fname)
    events = mne.find_events(raw)
    event_id = {
        'Auditory/Left': 1,
        'Auditory/Right': 2,
        'Visual/Left': 3,
        'Visual/Right': 4,
        'Visual/Smiley': 32,
        'Motor/Button': 5
    }
    event_map = {v: k for k, v in event_id.items()}
    annot = Annotations(onset=raw.times[events[:, 0] - raw.first_samp],
                        duration=np.zeros(len(events)),
                        description=[event_map[vv] for vv in events[:, 2]],
                        orig_time=None)
    raw.set_annotations(annot)

    events2, event_id2 = \
        events_from_annotations(raw, event_id=event_id, regexp=None)
    assert_array_equal(events, events2)
    assert_equal(event_id, event_id2)

    events3, event_id3 = \
        events_from_annotations(raw, event_id=None, regexp=None)

    assert_array_equal(events[:, 0], events3[:, 0])
    assert set(event_id.keys()) == set(event_id3.keys())

    # ensure that these actually got sorted properly
    expected_event_id = {
        desc: idx + 1 for idx, desc in enumerate(sorted(event_id.keys()))}
    assert event_id3 == expected_event_id

    first = np.unique(events3[:, 2])
    second = np.arange(1, len(event_id) + 1, 1).astype(first.dtype)
    assert_array_equal(first, second)

    first = np.unique(list(event_id3.values()))
    second = np.arange(1, len(event_id) + 1, 1).astype(first.dtype)
    assert_array_equal(first, second)

    events4, event_id4 =\
        events_from_annotations(raw, event_id=None, regexp='.*Left')

    expected_event_id4 = {k: v for k, v in event_id.items() if 'Left' in k}
    assert_equal(event_id4.keys(), expected_event_id4.keys())

    expected_events4 = events[(events[:, 2] == 1) | (events[:, 2] == 3)]
    assert_array_equal(expected_events4[:, 0], events4[:, 0])

    events5, event_id5 = \
        events_from_annotations(raw, event_id=event_id, regexp='.*Left')

    expected_event_id5 = {k: v for k, v in event_id.items() if 'Left' in k}
    assert_equal(event_id5, expected_event_id5)

    expected_events5 = events[(events[:, 2] == 1) | (events[:, 2] == 3)]
    assert_array_equal(expected_events5, events5)

    with pytest.raises(ValueError, match='not find any of the events'):
        events_from_annotations(raw, regexp='not_there')

    with pytest.raises(ValueError, match='Invalid type for event_id'):
        events_from_annotations(raw, event_id='wrong')

    # concat does not introduce BAD or EDGE
    raw_concat = concatenate_raws([raw.copy(), raw.copy()])
    _, event_id = events_from_annotations(raw_concat)
    assert isinstance(event_id, dict)
    assert len(event_id) > 0
    for kind in ('BAD', 'EDGE'):
        assert '%s boundary' % kind in raw_concat.annotations.description
        for key in event_id.keys():
            assert kind not in key

    # remove all events
    raw.set_annotations(None)
    events7, _ = events_from_annotations(raw)
    assert_array_equal(events7, np.empty((0, 3), dtype=int))


def test_events_from_annot_onset_alingment():
    """Test events and annotations onset are the same."""
    raw = _raw_annot(meas_date=1, orig_time=1.5)
    #       sec  0        1        2        3
    #       raw  .        |--------XXXXXXXXX
    #     annot  .             |---XX
    # raw.annot  .        |--------XX
    #   latency  .        0        1        2
    #            .                 0        0

    assert raw.annotations.orig_time == _handle_meas_date(1)
    assert raw.annotations.onset[0] == 1
    assert raw.first_samp == 10
    event_latencies, event_id = events_from_annotations(raw)
    assert event_latencies[0, 0] == 10
    assert raw.first_samp == event_latencies[0, 0]


def _create_annotation_based_on_descr(description, annotation_start_sampl=0,
                                      duration=0, orig_time=0):
    """Create a raw object with annotations from descriptions.

    The returning raw object contains as many annotations as description given.
    All starting at `annotation_start_sampl`.
    """
    # create dummy raw
    raw = RawArray(data=np.empty([10, 10], dtype=np.float64),
                   info=create_info(ch_names=10, sfreq=1000.),
                   first_samp=0)
    raw.set_meas_date(0)

    # create dummy annotations based on the descriptions
    onset = raw.times[annotation_start_sampl]
    onset_matching_desc = np.full_like(description, onset, dtype=type(onset))
    duration_matching_desc = np.full_like(description, duration,
                                          dtype=type(duration))
    annot = Annotations(description=description,
                        onset=onset_matching_desc,
                        duration=duration_matching_desc,
                        orig_time=orig_time)

    if duration != 0:
        with pytest.warns(RuntimeWarning, match='Limited.*expanding outside'):
            # duration 0.1s is larger than the raw data expand
            raw.set_annotations(annot)
    else:
        raw.set_annotations(annot)

    # Make sure that set_annotations(annot) works
    assert all(raw.annotations.onset == onset)
    if duration != 0:
        expected_duration = (len(raw.times) / raw.info['sfreq']) - onset
    else:
        expected_duration = 0
    _duration = raw.annotations.duration[0]
    assert _duration == approx(expected_duration)
    assert all(raw.annotations.duration == _duration)
    assert all(raw.annotations.description == description)

    return raw


def test_event_id_function_default():
    """Test[unit_test] for event_id_function default in event_from_annotations.

    The expected behavior is give numeric label for all those annotations not
    present in event_id, starting at 1.
    """
    # No event_id given
    description = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    expected_event_id = dict(zip(description, range(1, 100)))
    expected_events = np.array([[3, 3, 3, 3, 3, 3, 3],
                                [0, 0, 0, 0, 0, 0, 0],
                                [1, 2, 3, 4, 5, 6, 7]]).T

    raw = _create_annotation_based_on_descr(description,
                                            annotation_start_sampl=3,
                                            duration=100)
    events, event_id = events_from_annotations(raw, event_id=None)

    assert_array_equal(events, expected_events)
    assert event_id == expected_event_id


def test_event_id_function_using_custom_function():
    """Test [unit_test] arbitrary function to create the ids."""
    def _constant_id(*args, **kwargs):
        return 42

    description = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    expected_event_id = dict(zip(description, repeat(42)))
    expected_events = np.repeat([[0, 0, 42]], len(description), axis=0)
    raw = _create_annotation_based_on_descr(description)
    events, event_id = events_from_annotations(raw, event_id=_constant_id)

    assert_array_equal(events, expected_events)
    assert event_id == expected_event_id


# Test for IO with .csv files


def _assert_annotations_equal(a, b, tol=0):
    __tracebackhide__ = True
    assert_allclose(a.onset, b.onset, rtol=0, atol=tol)
    assert_allclose(a.duration, b.duration, rtol=0, atol=tol)
    assert_array_equal(a.description, b.description)
    a_orig_time = a.orig_time
    b_orig_time = b.orig_time
    assert a_orig_time == b_orig_time


@pytest.fixture(scope='session')
def dummy_annotation_csv_file(tmpdir_factory):
    """Create csv file for testing."""
    content = ("onset,duration,description\n"
               "2002-12-03 19:01:11.720100,1.0,AA\n"
               "2002-12-03 19:01:20.720100,2.425,BB")

    fname = tmpdir_factory.mktemp('data').join('annotations.csv')
    fname.write(content)
    return fname


@pytest.fixture(scope='session')
def dummy_broken_annotation_csv_file(tmpdir_factory):
    """Create csv file for testing."""
    content = ("onset,duration,description\n"
               "1.,1.0,AA\n"
               "3.,2.425,BB")

    fname = tmpdir_factory.mktemp('data').join('annotations_broken.csv')
    fname.write(content)
    return fname


@requires_version('pandas', '0.16')
def test_io_annotation_csv(dummy_annotation_csv_file,
                           dummy_broken_annotation_csv_file,
                           tmpdir_factory):
    """Test CSV input/output."""
    annot = read_annotations(str(dummy_annotation_csv_file))
    assert annot.orig_time == _handle_meas_date(1038942071.7201)
    assert_array_equal(annot.onset, np.array([0., 9.], dtype=np.float32))
    assert_array_almost_equal(annot.duration, [1., 2.425])
    assert_array_equal(annot.description, ['AA', 'BB'])

    # Now test writing
    fname = str(tmpdir_factory.mktemp('data').join('annotations.csv'))
    annot.save(fname)
    annot2 = read_annotations(fname)
    _assert_annotations_equal(annot, annot2)

    # Now without an orig_time
    annot._orig_time = None
    annot.save(fname)
    annot2 = read_annotations(fname)
    _assert_annotations_equal(annot, annot2)

    # Test broken .csv that does not use timestamps
    with pytest.warns(RuntimeWarning, match='save your CSV as a TXT'):
        annot2 = read_annotations(str(dummy_broken_annotation_csv_file))


# Test for IO with .txt files

@pytest.fixture(scope='session')
def dummy_annotation_txt_file(tmpdir_factory):
    """Create txt file for testing."""
    content = ("3.14, 42, AA \n"
               "6.28, 48, BB")

    fname = tmpdir_factory.mktemp('data').join('annotations.txt')
    fname.write(content)
    return fname


def test_io_annotation_txt(dummy_annotation_txt_file, tmpdir_factory):
    """Test TXT input/output."""
    annot = read_annotations(str(dummy_annotation_txt_file))
    assert annot.orig_time is None
    assert_array_equal(annot.onset, [3.14, 6.28])
    assert_array_equal(annot.duration, [42., 48])
    assert_array_equal(annot.description, ['AA', 'BB'])

    # Now test writing
    fname = str(tmpdir_factory.mktemp('data').join('annotations.txt'))
    annot.save(fname)
    annot2 = read_annotations(fname)
    _assert_annotations_equal(annot, annot2)

    # Now with an orig_time
    assert annot.orig_time is None
    annot._orig_time = _handle_meas_date(1038942071.7201)
    assert annot.orig_time is not None
    annot.save(fname)
    annot2 = read_annotations(fname)
    assert annot2.orig_time is not None
    _assert_annotations_equal(annot, annot2)


@pytest.fixture(scope='session')
def dummy_annotation_txt_header(tmpdir_factory):
    """Create txt header."""
    content = ("# A something \n"
               "# orig_time : 42\n"
               "# orig_time : 2002-12-03 19:01:11.720100\n"
               "# orig_time : 42\n"
               "# C\n"
               "Done")
    fname = tmpdir_factory.mktemp('data').join('header.txt')
    fname.write(content)
    return str(fname)


@pytest.mark.parametrize('meas_date, out', [
    pytest.param('toto', None, id='invalid string'),
    pytest.param(None, None, id='None'),
    pytest.param(42, 42.0, id='Scalar'),
    pytest.param(3.14, 3.14, id='Float'),
    pytest.param((3, 140000), 3.14, id='Scalar touple'),
    pytest.param('2002-12-03 19:01:11.720100', 1038942071.7201,
                 id='valid iso8601 string'),
    pytest.param('2002-12-03T19:01:11.720100', None,
                 id='invalid iso8601 string')])
def test_handle_meas_date(meas_date, out):
    """Test meas date formats."""
    if out is not None:
        assert out >= 0  # otherwise it'll break on Windows
        out = datetime.fromtimestamp(out, timezone.utc)
    assert _handle_meas_date(meas_date) == out


def test_read_annotation_txt_header(dummy_annotation_txt_header):
    """Test TXT orig_time recovery."""
    orig_time = _read_annotations_txt_parse_header(dummy_annotation_txt_header)
    want = datetime.fromtimestamp(1038942071.7201, timezone.utc)
    assert orig_time == want


@pytest.fixture(scope='session')
def dummy_annotation_txt_file_with_orig_time(tmpdir_factory):
    """Create TXT annotations with header."""
    content = ("# MNE-Annotations\n"
               "# orig_time : 2002-12-03 19:01:11.720100\n"
               "# onset, duration, description\n"
               "3.14, 42, AA \n"
               "6.28, 48, BB")

    fname = tmpdir_factory.mktemp('data').join('annotations.txt')
    fname.write(content)
    return fname


def test_read_annotation_txt_orig_time(
        dummy_annotation_txt_file_with_orig_time):
    """Test TXT input/output."""
    annot = read_annotations(str(dummy_annotation_txt_file_with_orig_time))
    dt = datetime.fromtimestamp(1038942071.7201, timezone.utc)
    want = Annotations([3.14, 6.28], [42., 48], ['AA', 'BB'], dt)
    assert annot == want


@pytest.fixture(scope='session')
def dummy_annotation_txt_one_segment(tmpdir_factory):
    """Create empty TXT annotations."""
    content = ("# MNE-Annotations\n"
               "# onset, duration, description\n"
               "3.14, 42, AA")
    fname = tmpdir_factory.mktemp('data').join('one-annotations.txt')
    fname.write(content)
    return fname


def test_read_annotation_txt_one_segment(
        dummy_annotation_txt_one_segment):
    """Test empty TXT input/output."""
    annot = read_annotations(str(dummy_annotation_txt_one_segment))
    assert_array_equal(annot.onset, 3.14)
    assert_array_equal(annot.duration, 42)
    assert_array_equal(annot.description, 'AA')


@pytest.fixture(scope='session')
def dummy_annotation_txt_file_empty(tmpdir_factory):
    """Create empty TXT annotations."""
    content = ("# MNE-Annotations\n"
               "# onset, duration, description\n")
    fname = tmpdir_factory.mktemp('data').join('empty-annotations.txt')
    fname.write(content)
    return fname


def test_read_annotation_txt_empty(
        dummy_annotation_txt_file_empty):
    """Test empty TXT input/output."""
    annot = read_annotations(str(dummy_annotation_txt_file_empty))
    assert_array_equal(annot.onset, np.array([], dtype=np.float64))
    assert_array_equal(annot.duration, np.array([], dtype=np.float64))
    assert_array_equal(annot.description, np.array([], dtype='<U1'))


def test_annotations_simple_iteration():
    """Test indexing Annotations."""
    NUM_ANNOT = 5
    EXPECTED_ELEMENTS_TYPE = (np.float64, np.float64, np.str_)
    EXPECTED_ONSETS = EXPECTED_DURATIONS = [x for x in range(NUM_ANNOT)]
    EXPECTED_DESCS = [x.__repr__() for x in range(NUM_ANNOT)]

    annot = Annotations(onset=EXPECTED_ONSETS,
                        duration=EXPECTED_DURATIONS,
                        description=EXPECTED_DESCS,
                        orig_time=None)

    for ii, elements in enumerate(annot[:2]):
        assert isinstance(elements, OrderedDict)
        expected_values = (ii, ii, str(ii))
        for elem, expected_type, expected_value in zip(elements.values(),
                                                       EXPECTED_ELEMENTS_TYPE,
                                                       expected_values):
            assert np.isscalar(elem)
            assert type(elem) == expected_type
            assert elem == expected_value


@requires_version('numpy', '1.12')
def test_annotations_slices():
    """Test indexing Annotations."""
    NUM_ANNOT = 5
    EXPECTED_ONSETS = EXPECTED_DURATIONS = [x for x in range(NUM_ANNOT)]
    EXPECTED_DESCS = [x.__repr__() for x in range(NUM_ANNOT)]

    annot = Annotations(onset=EXPECTED_ONSETS,
                        duration=EXPECTED_DURATIONS,
                        description=EXPECTED_DESCS,
                        orig_time=None)

    # Indexing returns a copy. So this has no effect in annot
    annot[0]['onset'] = 42
    annot[0]['duration'] = 3.14
    annot[0]['description'] = 'foobar'

    annot[:1].onset[0] = 42
    annot[:1].duration[0] = 3.14
    annot[:1].description[0] = 'foobar'

    # Slicing with single element returns a dictionary
    for ii in EXPECTED_ONSETS:
        assert annot[ii] == dict(zip(['onset', 'duration',
                                      'description', 'orig_time'],
                                     [ii, ii, str(ii), None]))

    # Slices should give back Annotations
    for current in (annot[slice(0, None, 2)],
                    annot[[bool(ii % 2) for ii in range(len(annot))]],
                    annot[:1],
                    annot[[0, 2, 2]],
                    annot[(0, 2, 2)],
                    annot[np.array([0, 2, 2])],
                    annot[1::2],
                    ):
        assert isinstance(current, Annotations)
        assert len(current) != len(annot)

    for bad_ii in [len(EXPECTED_ONSETS), 42, 'foo']:
        with pytest.raises(IndexError):
            annot[bad_ii]


def test_sorting():
    """Test annotation sorting."""
    annot = Annotations([10, 20, 30], [1, 2, 3], 'BAD')
    # assert_array_equal(annot.onset, [0, 5, 10])
    annot.append([5, 15, 25, 35], 0.5, 'BAD')
    onset = list(range(5, 36, 5))
    duration = list(annot.duration)
    assert_array_equal(annot.onset, onset)
    assert_array_equal(annot.duration, duration)
    annot.append([10, 10], [0.1, 9], 'BAD')  # 0.1 should be before, 9 after
    want_before = onset.index(10)
    duration.insert(want_before, 0.1)
    duration.insert(want_before + 2, 9)
    onset.insert(want_before, 10)
    onset.insert(want_before, 10)
    assert_array_equal(annot.onset, onset)
    assert_array_equal(annot.duration, duration)


def test_date_none(tmpdir):
    """Test that DATE_NONE is used properly."""
    # Regression test for gh-5908
    n_chans = 139
    n_samps = 20
    data = np.random.random_sample((n_chans, n_samps))
    ch_names = ['E{}'.format(x) for x in range(n_chans)]
    ch_types = ['eeg'] * n_chans
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=2048)
    assert info['meas_date'] is None
    raw = RawArray(data=data, info=info)
    fname = op.join(str(tmpdir), 'test-raw.fif')
    raw.save(fname)
    raw_read = read_raw_fif(fname, preload=True)
    assert raw_read.info['meas_date'] is None


def test_negative_meas_dates(windows_like_datetime):
    """Test meas_date previous to 1970."""
    # Regression test for gh-6621
    raw = RawArray(data=np.empty((1, 1), dtype=np.float64),
                   info=create_info(ch_names=1, sfreq=1.))
    raw.set_meas_date((-908196946, 988669))
    raw.set_annotations(Annotations(description='foo', onset=[0],
                                    duration=[0], orig_time=None))
    events, _ = events_from_annotations(raw)
    assert events[:, 0] == 0


def test_crop_when_negative_orig_time(windows_like_datetime):
    """Test cropping with orig_time, tmin and tmax previous to 1970."""
    # Regression test for gh-6621
    orig_time_stamp = -908196945.011331  # 1941-03-22 11:04:14.988669
    annot = Annotations(description='foo', onset=np.arange(0, 0.999, 0.1),
                        duration=[0], orig_time=orig_time_stamp)
    stamp = _dt_to_stamp(annot.orig_time)
    assert_allclose(stamp[0] + stamp[1] * 1e-6, orig_time_stamp)
    t = stamp[0] + stamp[1] * 1e-6
    assert t == orig_time_stamp
    assert len(annot) == 10

    # do not raise
    annot.crop(verbose='debug')
    assert len(annot) == 10

    # Crop with negative tmin, tmax
    tmin, tmax = [orig_time_stamp + t for t in (0.25, .75)]
    assert tmin < 0 and tmax < 0
    crop_annot = annot.crop(tmin=tmin, tmax=tmax)
    assert_allclose(crop_annot.onset, [0.3, 0.4, 0.5, 0.6, 0.7])
    orig_dt = _stamp_to_dt(stamp)
    assert crop_annot.orig_time == orig_dt  # orig_time does not change


def test_allow_nan_durations():
    """Deal with "n/a" strings in BIDS events with nan durations."""
    raw = RawArray(data=np.empty([2, 10], dtype=np.float64),
                   info=create_info(ch_names=2, sfreq=1.),
                   first_samp=0)
    raw.set_meas_date(0)

    ons = [1, 2., 15., 17.]
    dus = [np.nan, 1., 0.5, np.nan]
    descriptions = ['A'] * 4
    onsets = np.asarray(ons, dtype=float)
    durations = np.asarray(dus, dtype=float)
    annot = mne.Annotations(onset=onsets,
                            duration=durations,
                            description=descriptions)
    with pytest.warns(RuntimeWarning, match='Omitted 2 annotation'):
        raw.set_annotations(annot)


@testing.requires_testing_data
def test_annotations_from_events():
    """Test events to annotations conversion."""
    raw = read_raw_fif(fif_fname)
    events = mne.find_events(raw)

    # 1. Automatic event description
    # -------------------------------------------------------------------------
    annots = annotations_from_events(events, raw.info['sfreq'],
                                     first_samp=raw.first_samp,
                                     orig_time=None)
    assert len(annots) == events.shape[0]

    # Convert back to events
    raw.set_annotations(annots)
    events_out, _ = events_from_annotations(raw, event_id=int)
    assert_array_equal(events, events_out)

    # 2. Explicit event mapping
    # -------------------------------------------------------------------------
    event_desc = {1: 'one', 2: 'two', 3: 'three', 32: None}
    annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                     event_desc=event_desc,
                                     first_samp=raw.first_samp,
                                     orig_time=None)

    assert np.all([a in ['one', 'two', 'three'] for a in annots.description])
    assert len(annots) == events[events[:, 2] <= 3].shape[0]

    # 3. Pass list
    # -------------------------------------------------------------------------
    event_desc = [1, 2, 3]
    annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                     event_desc=event_desc,
                                     first_samp=raw.first_samp,
                                     orig_time=None)

    assert np.all([a in ['1', '2', '3'] for a in annots.description])
    assert len(annots) == events[events[:, 2] <= 3].shape[0]

    # 4. Try passing callable
    # -------------------------------------------------------------------------
    event_desc = lambda d: 'event{}'.format(d)  # noqa:E731
    annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                     event_desc=event_desc,
                                     first_samp=raw.first_samp,
                                     orig_time=None)

    assert np.all(['event' in a for a in annots.description])
    assert len(annots) == events.shape[0]

    # 5. Pass numpy array
    # -------------------------------------------------------------------------
    event_desc = np.array([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError, match='event_desc must be 1D'):
        annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                         event_desc=event_desc,
                                         first_samp=raw.first_samp,
                                         orig_time=None)

    with pytest.raises(ValueError, match='Invalid type for event_desc'):
        annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                         event_desc=1,
                                         first_samp=raw.first_samp,
                                         orig_time=None)

    event_desc = np.array([1, 2, 3])
    annots = annotations_from_events(events, sfreq=raw.info['sfreq'],
                                     event_desc=event_desc,
                                     first_samp=raw.first_samp,
                                     orig_time=None)
    assert np.all([a in ['1', '2', '3'] for a in annots.description])
    assert len(annots) == events[events[:, 2] <= 3].shape[0]


run_tests_if_main()
