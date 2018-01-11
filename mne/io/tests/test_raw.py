# Generic tests that all raw classes should run
from os import path as op
import math
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_equal, assert_true

from mne import concatenate_raws
from mne.datasets import testing
from mne.io import read_raw_fif
from mne import find_events, Annotations
from mne.utils import _TempDir


def _test_raw_reader(reader, test_preloading=True, **kwargs):
    """Test reading, writing and slicing of raw classes.

    Parameters
    ----------
    reader : function
        Function to test.
    test_preloading : bool
        Whether not preloading is implemented for the reader. If True, both
        cases and memory mapping to file are tested.
    **kwargs :
        Arguments for the reader. Note: Do not use preload as kwarg.
        Use ``test_preloading`` instead.

    Returns
    -------
    raw : Instance of Raw
        A preloaded Raw object.
    """
    tempdir = _TempDir()
    rng = np.random.RandomState(0)
    if test_preloading:
        raw = reader(preload=True, **kwargs)
        # don't assume the first is preloaded
        buffer_fname = op.join(tempdir, 'buffer')
        picks = rng.permutation(np.arange(len(raw.ch_names) - 1))[:10]
        picks = np.append(picks, len(raw.ch_names) - 1)  # test trigger channel
        bnd = min(int(round(raw.info['buffer_size_sec'] *
                            raw.info['sfreq'])), raw.n_times)
        slices = [slice(0, bnd), slice(bnd - 1, bnd), slice(3, bnd),
                  slice(3, 300), slice(None), slice(1, bnd)]
        if raw.n_times >= 2 * bnd:  # at least two complete blocks
            slices += [slice(bnd, 2 * bnd), slice(bnd, bnd + 1),
                       slice(0, bnd + 100)]
        other_raws = [reader(preload=buffer_fname, **kwargs),
                      reader(preload=False, **kwargs)]
        for sl_time in slices:
            data1, times1 = raw[picks, sl_time]
            for other_raw in other_raws:
                data2, times2 = other_raw[picks, sl_time]
                assert_allclose(data1, data2)
                assert_allclose(times1, times2)
    else:
        raw = reader(**kwargs)

    full_data = raw._data
    assert_true(raw.__class__.__name__, repr(raw))  # to test repr
    assert_true(raw.info.__class__.__name__, repr(raw.info))

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_raw.fif')
    raw = concatenate_raws([raw])
    raw.save(out_fname, tmax=raw.times[-1], overwrite=True, buffer_size_sec=1)
    raw3 = read_raw_fif(out_fname)
    assert_equal(set(raw.info.keys()), set(raw3.info.keys()))
    assert_allclose(raw3[0:20][0], full_data[0:20], rtol=1e-6,
                    atol=1e-20)  # atol is very small but > 0
    assert_array_almost_equal(raw.times, raw3.times)

    assert_true(not math.isnan(raw3.info['highpass']))
    assert_true(not math.isnan(raw3.info['lowpass']))
    assert_true(not math.isnan(raw.info['highpass']))
    assert_true(not math.isnan(raw.info['lowpass']))

    assert_equal(raw3.info['kit_system_id'], raw.info['kit_system_id'])

    # Make sure concatenation works
    first_samp = raw.first_samp
    last_samp = raw.last_samp
    concat_raw = concatenate_raws([raw.copy(), raw])
    assert_equal(concat_raw.n_times, 2 * raw.n_times)
    assert_equal(concat_raw.first_samp, first_samp)
    assert_equal(concat_raw.last_samp - last_samp + first_samp, last_samp + 1)
    idx = np.where(concat_raw.annotations.description == 'BAD boundary')[0]
    assert_array_almost_equal([(last_samp - first_samp) / raw.info['sfreq']],
                              concat_raw.annotations.onset[idx], decimal=2)
    return raw


def _test_concat(reader, *args):
    """Test concatenation of raw classes that allow not preloading."""
    data = None

    for preload in (True, False):
        raw1 = reader(*args, preload=preload)
        raw2 = reader(*args, preload=preload)
        raw1.append(raw2)
        raw1.load_data()
        if data is None:
            data = raw1[:, :][0]
        assert_allclose(data, raw1[:, :][0])

    for first_preload in (True, False):
        raw = reader(*args, preload=first_preload)
        data = raw[:, :][0]
        for preloads in ((True, True), (True, False), (False, False)):
            for last_preload in (True, False):
                t_crops = raw.times[np.argmin(np.abs(raw.times - 0.5)) +
                                    [0, 1]]
                raw1 = raw.copy().crop(0, t_crops[0])
                if preloads[0]:
                    raw1.load_data()
                raw2 = raw.copy().crop(t_crops[1], None)
                if preloads[1]:
                    raw2.load_data()
                raw1.append(raw2)
                if last_preload:
                    raw1.load_data()
                assert_allclose(data, raw1[:, :][0])


@testing.requires_testing_data
def test_time_index():
    """Test indexing of raw times."""
    raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                        'data', 'test_raw.fif')
    raw = read_raw_fif(raw_fname)

    # Test original (non-rounding) indexing behavior
    orig_inds = raw.time_as_index(raw.times)
    assert(len(set(orig_inds)) != len(orig_inds))

    # Test new (rounding) indexing behavior
    new_inds = raw.time_as_index(raw.times, use_rounding=True)
    assert(len(set(new_inds)) == len(new_inds))


@testing.requires_testing_data
def test_events_from_annot():
    raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                        'data', 'test_raw.fif')
    raw = read_raw_fif(raw_fname)
    events = find_events(raw)
    event_id = {
        'Auditory/Left': 1,
        'Auditory/Right': 2,
        'Visual/Left': 3,
        'Visual/Right': 4,
        'Visual/Smiley': 32,
        'Motor/Button': 5
    }
    event_map = {v: k for k, v in event_id.items()}
    raw.annotations = Annotations(
        onset=raw.times[events[:, 0] - raw.first_samp],
        duration=np.zeros(len(events)),
        description=[event_map[vv] for vv in events[:, 2]])

    events2, event_id2 = raw.events_from_annot(
        event_id=event_id, regexp=None)
    assert_array_equal(events, events2)
    assert_equal(event_id, event_id2)

    events3, event_id3 = raw.events_from_annot(
        event_id=None, regexp=None)

    assert_array_equal(events[:, 0], events3[:, 0])
    assert_equal(event_id.keys(), event_id3.keys())

    first = np.unique(events3[:, 2])
    second = np.arange(1, len(event_id) + 1, 1).astype(first.dtype)
    assert_array_equal(first, second)

    first = np.unique(list(event_id3.values()))
    second = np.arange(1, len(event_id) + 1, 1).astype(first.dtype)
    assert_array_equal(first, second)

    events4, event_id4 = raw.events_from_annot(
        event_id=None, regexp='.*Left')

    expected_event_id4 = {k: v for k, v in event_id.items() if 'Left' in k}
    assert_equal(event_id4.keys(), expected_event_id4.keys())

    expected_events4 = events[(events[:, 2] == 1) | (events[:, 2] == 3)]
    assert_array_equal(expected_events4[:, 0], events4[:, 0])

    events5, event_id5 = raw.events_from_annot(
        event_id=event_id, regexp='.*Left')

    expected_event_id5 = {k: v for k, v in event_id.items() if 'Left' in k}
    assert_equal(event_id5, expected_event_id5)

    expected_events5 = events[(events[:, 2] == 1) | (events[:, 2] == 3)]
    assert_array_equal(expected_events5, events5)
