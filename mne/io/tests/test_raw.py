# Generic tests that all raw classes should run
from os import path as op
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)
from nose.tools import assert_equal

from mne.datasets import testing
from mne.io import Raw
from mne.utils import _TempDir


def _test_raw_object(reader, test_preloading, **kwargs):
    """Test reading, writing and slicing of raw classes.

    Parameters
    ----------
    reader : function
        Function to test.
    test_preloading : bool
        Whether not preloading is implemented for the reader. If True, both
        cases and memory mapping to file are tested.
    **kwargs :
        Arguments for the reader.

    Returns
    -------
    raw : Instance of Raw
        A preloaded Raw object.
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

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_raw.fif')
    for obj in raws:
        obj.save(out_fname, tmax=obj.times[-1], overwrite=True)
        raw3 = Raw(out_fname)
        assert_equal(sorted(raw.info.keys()), sorted(raw3.info.keys()))
        assert_array_almost_equal(raw3.load_data()._data[0:20],
                                  full_data[0:20])

    return raw


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
