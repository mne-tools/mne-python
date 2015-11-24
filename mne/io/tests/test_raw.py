# Generic tests that all raw classes should run
import warnings
from os import path as op
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)
from nose.tools import assert_equal

from mne.datasets import testing
from mne.io import Raw, base
from mne.utils import _TempDir
from mne.io.pick import _pick_data_channels


def _test_raw_object(reader, test_preloading, **kwargs):
    """Test reading, writing and concatenating of raw classes.

    Parameters
    ----------
    reader : function
        Function to test.
    test_preloading : bool
        Whether not preloading is implemented for the reader. If True, both
        cases are tested.
    **kwargs :
        Arguments for the reader.
    """
    tempdir = _TempDir()
    raws = list()
    raws.append(reader(**kwargs))
    if test_preloading:
        buffer_fname = op.join(tempdir, 'buffer')
        raws.append(reader(preload=buffer_fname, **kwargs))
        raws.append(reader(preload=True, **kwargs))
        picks = [1, 3, 5]
        assert_array_equal(raws[0][picks, 20:30][0], raws[-1][picks, 20:30][0])
    raw = raws[-1]  # use preloaded raw
    full_data = raw._data

    print(raw)  # to test repr
    print(raw.info)  # to test Info reprs

    # Test concatenation
    raw2 = base.concatenate_raws([raw.copy(), raw])
    assert_equal(raw2.n_times, 2 * raw.n_times)

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_raw.fif')
    for obj in raws:
        obj.save(out_fname, tmax=obj.times[-1], overwrite=True)
        raw3 = Raw(out_fname)
        assert_equal(sorted(raw.info.keys()), sorted(raw3.info.keys()))
        assert_array_almost_equal(raw3.load_data()._data[0:20],
                                  full_data[0:20])

    data1, times1 = raw[:10:3, 10:12]
    data2, times2 = raw2[:10:3, 10:12]
    data3, times3 = raw2[[0, 3, 6, 9], 10:12]
    assert_array_almost_equal(data1, full_data[:10:3, 10:12], 9)
    assert_array_almost_equal(data1, data2, 9)
    assert_array_almost_equal(data1, data3, 9)
    assert_array_almost_equal(times1, times2)
    assert_array_almost_equal(times1, times3)
    return raw  # raw object to feed for filter test


def _test_raw_filter(raw, atol=0):
    """Test filtering of raw classes.

    Parameters
    ----------
    raw : instance of Raw
        Raw object to test.
    atol : float
        Absolute tolerance added for the tests.
        Comparison to: ``atol + 0.015 * abs(desired)``. 0 by default.
    """
    picks = _pick_data_channels(raw.info)[:4]
    assert_equal(len(picks), 4)
    raw_lp = raw.copy()
    with warnings.catch_warnings(record=True):
        raw_lp.filter(0., 4.0 - 0.25, picks=picks, n_jobs=2)
    raw_hp = raw.copy()
    with warnings.catch_warnings(record=True):
        raw_hp.filter(8.0 + 0.25, None, picks=picks, n_jobs=2)
    raw_bp = raw.copy()
    with warnings.catch_warnings(record=True):
        raw_bp.filter(4.0 + 0.25, 8.0 - 0.25, picks=picks)
    raw_bs = raw.copy()
    with warnings.catch_warnings(record=True):
        raw_bs.filter(8.0 + 0.25, 4.0 - 0.25, picks=picks, n_jobs=2)
    data, _ = raw[picks, :]
    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]

    assert_allclose(data, lp_data + bp_data + hp_data, 0.015, atol=atol)
    assert_allclose(data, bp_data + bs_data, 0.015, atol=atol)


def _test_concat(reader, *args):
    """Test concatenation of raw classes that allow not preloading"""
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
                print(first_preload, preloads, last_preload)
                raw1 = raw.crop(0, 0.4999)
                if preloads[0]:
                    raw1.load_data()
                raw2 = raw.crop(0.5, None)
                if preloads[1]:
                    raw2.load_data()
                raw1.append(raw2)
                if last_preload:
                    raw1.load_data()
                assert_allclose(data, raw1[:, :][0])


@testing.requires_testing_data
def test_time_index():
    """Test indexing of raw times"""
    raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                        'data', 'test_raw.fif')
    raw = Raw(raw_fname)

    # Test original (non-rounding) indexing behavior
    orig_inds = raw.time_as_index(raw.times)
    assert(len(set(orig_inds)) != len(orig_inds))

    # Test new (rounding) indexing behavior
    new_inds = raw.time_as_index(raw.times, use_rounding=True)
    assert(len(set(new_inds)) == len(new_inds))
