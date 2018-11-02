# -*- coding: utf-8 -*-
"""Generic tests that all raw classes should run."""
# # Authors: MNE Developers
#            Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

from os import path as op
import math

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_equal, assert_array_equal)

from mne import concatenate_raws, create_info, Annotations
from mne.annotations import _handle_meas_date
from mne.datasets import testing
from mne.io import read_raw_fif, RawArray, BaseRaw
from mne.utils import _TempDir
from mne.io.meas_info import _get_valid_units


def test_orig_units():
    """Test the error handling for original units."""
    # Should work fine
    info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
    BaseRaw(info, last_samps=[1], orig_units={'Cz': 'nV'})

    # Should complain that channel Cz does not have a corresponding original
    # unit.
    with pytest.raises(ValueError, match='has no associated original unit.'):
        info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
        BaseRaw(info, last_samps=[1], orig_units={'not_Cz': 'nV'})

    # Test that a non-dict orig_units argument raises a ValueError
    with pytest.raises(ValueError, match='orig_units must be of type dict'):
        info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
        BaseRaw(info, last_samps=[1], orig_units=True)


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
        bnd = min(int(round(raw.buffer_size_sec *
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
    assert raw.__class__.__name__ in repr(raw)  # to test repr
    assert raw.info.__class__.__name__ in repr(raw.info)

    # gh-5604
    assert _handle_meas_date(raw.info['meas_date']) >= 0

    # test resetting raw
    raw2 = reader(**raw._init_kwargs)
    assert set(raw.info.keys()) == set(raw2.info.keys())
    assert_array_equal(raw.times, raw2.times)

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_raw.fif')
    raw = concatenate_raws([raw])
    raw.save(out_fname, tmax=raw.times[-1], overwrite=True, buffer_size_sec=1)
    raw3 = read_raw_fif(out_fname)
    assert set(raw.info.keys()) == set(raw3.info.keys())
    assert_allclose(raw3[0:20][0], full_data[0:20], rtol=1e-6,
                    atol=1e-20)  # atol is very small but > 0
    assert_array_almost_equal(raw.times, raw3.times)

    assert not math.isnan(raw3.info['highpass'])
    assert not math.isnan(raw3.info['lowpass'])
    assert not math.isnan(raw.info['highpass'])
    assert not math.isnan(raw.info['lowpass'])

    assert raw3.info['kit_system_id'] == raw.info['kit_system_id']

    # Make sure concatenation works
    first_samp = raw.first_samp
    last_samp = raw.last_samp
    concat_raw = concatenate_raws([raw.copy(), raw])
    assert_equal(concat_raw.n_times, 2 * raw.n_times)
    assert_equal(concat_raw.first_samp, first_samp)
    assert_equal(concat_raw.last_samp - last_samp + first_samp, last_samp + 1)
    idx = np.where(concat_raw.annotations.description == 'BAD boundary')[0]

    if concat_raw.info['meas_date'] is None:
        expected_bad_boundary_onset = ((last_samp - first_samp) /
                                       raw.info['sfreq'])
    else:
        expected_bad_boundary_onset = raw._last_time

    assert_array_almost_equal(concat_raw.annotations.onset[idx],
                              expected_bad_boundary_onset,
                              decimal=2)

    if raw.info['meas_id'] is not None:
        for key in ['secs', 'usecs', 'version']:
            assert_equal(raw.info['meas_id'][key], raw3.info['meas_id'][key])
        assert_array_equal(raw.info['meas_id']['machid'],
                           raw3.info['meas_id']['machid'])

    assert isinstance(raw.annotations, Annotations)

    # Make a "soft" test on units: They have to be valid SI units as in
    # mne.io.meas_info.valid_units, but we accept any lower/upper case for now.
    valid_units = _get_valid_units()
    valid_units_lower = [unit.lower() for unit in valid_units]
    if raw._orig_units is not None:
        assert isinstance(raw._orig_units, dict)
        for ch_name, unit in raw._orig_units.items():
            assert unit.lower() in valid_units_lower, ch_name

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
def test_time_as_index():
    """Test indexing of raw times."""
    raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                        'data', 'test_raw.fif')
    raw = read_raw_fif(raw_fname)

    # Test original (non-rounding) indexing behavior
    orig_inds = raw.time_as_index(raw.times)
    assert(len(set(orig_inds)) != len(orig_inds))

    # Test new (rounding) indexing behavior
    new_inds = raw.time_as_index(raw.times, use_rounding=True)
    assert_array_equal(new_inds, np.arange(len(raw.times)))


@pytest.mark.parametrize('offset, origin', [
    pytest.param(0, None, id='times in s. relative to first_samp (default)'),
    pytest.param(0, 2.0, id='times in s. relative to first_samp'),
    pytest.param(1, 1.0, id='times in s. relative to meas_date'),
    pytest.param(2, 0.0, id='absolute times in s. relative to 0')])
def test_time_as_index_ref(offset, origin):
    """Test indexing of raw times."""
    meas_date = 1
    info = create_info(ch_names=10, sfreq=10.)
    raw = RawArray(data=np.empty((10, 10)), info=info, first_samp=10)
    raw.info['meas_date'] = meas_date

    relative_times = raw.times
    inds = raw.time_as_index(relative_times + offset,
                             use_rounding=True,
                             origin=origin)
    assert_array_equal(inds, np.arange(raw.n_times))


def test_annotation_property_deprecation_warning():
    """Test that assigning annotations warns and nowhere else."""
    with pytest.warns(None) as w:
        raw = RawArray(np.random.rand(1, 1), create_info(1, 1))
    assert len(w) is 0
    with pytest.warns(DeprecationWarning, match='by assignment is deprecated'):
        raw.annotations = None


def _raw_annot(meas_date, orig_time, sync_orig=True):
    info = create_info(ch_names=10, sfreq=10.)
    raw = RawArray(data=np.empty((10, 10)), info=info, first_samp=10)
    raw.info['meas_date'] = meas_date
    annot = Annotations([.5], [.2], ['dummy'], orig_time)
    raw.set_annotations(annotations=annot, sync_orig=sync_orig)
    return raw


def test_meas_date_orig_time():
    """Test the relation between meas_time in orig_time."""
    # meas_time is set and orig_time is set:
    # clips the annotations based on raw.data and resets the annotation based
    # on raw.info['meas_date]
    raw = _raw_annot(1, 1.5)
    assert raw.annotations.orig_time == 1
    assert raw.annotations.onset[0] == 1

    # meas_time is set and orig_time is None:
    # Consider annot.orig_time to be raw.frist_sample, clip and reset
    # annotations to have the raw.annotations.orig_time == raw.info['meas_date]
    raw = _raw_annot(1, None)
    assert raw.annotations.orig_time == 1
    assert raw.annotations.onset[0] == 1.5

    # meas_time is None and orig_time is set:
    # Raise error, it makes no sense to have an annotations object that we know
    # when was acquired and set it to a raw object that does not know when was
    # it acquired.
    with pytest.raises(RuntimeError, match='Ambiguous operation'):
        _raw_annot(None, 1.5)

    # meas_time is None and orig_time is None:
    # Consider annot.orig_time to be raw.first_sample and clip
    raw = _raw_annot(None, None)
    assert raw.annotations.orig_time is None
    assert raw.annotations.onset[0] == 0.5
    assert raw.annotations.duration[0] == 0.2


def test_deprecated_meas_date_orig_time():
    """Test meas_date_orig_time old behavior for backward compatibility."""
    with pytest.warns(DeprecationWarning):
        raw = _raw_annot(1, 1.5, sync_orig=False)
    assert raw.annotations.orig_time == 1.5
    assert raw.annotations.onset[0] == 0.5

    with pytest.warns(DeprecationWarning):
        raw = _raw_annot(1, None, sync_orig=False)
    assert raw.annotations.orig_time == 2
    assert raw.annotations.onset[0] == 0.5
