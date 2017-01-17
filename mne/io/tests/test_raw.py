# Generic tests that all raw classes should run
from os import path as op
import math
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from nose.tools import assert_equal, assert_true

from mne import concatenate_raws
from mne.datasets import testing
from mne.io import read_raw_fif
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
