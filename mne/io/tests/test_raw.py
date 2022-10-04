# -*- coding: utf-8 -*-
"""Generic tests that all raw classes should run."""
# Authors: MNE Developers
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

from contextlib import redirect_stdout
from io import StringIO
import math
import os
from os import path as op
from pathlib import Path
import re

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal, assert_array_less)

import mne
from mne import concatenate_raws, create_info, Annotations, pick_types
from mne.datasets import testing
from mne.io import read_raw_fif, RawArray, BaseRaw, Info, _writing_info_hdf5
from mne.io._digitization import _dig_kind_dict
from mne.io.base import _get_scaling
from mne.io.pick import _ELECTRODE_CH_TYPES, _FNIRS_CH_TYPES_SPLIT
from mne.utils import (_TempDir, catch_logging, _raw_annot, _stamp_to_dt,
                       object_diff, check_version, requires_pandas,
                       _import_h5io_funcs)
from mne.io.meas_info import _get_valid_units
from mne.io._digitization import DigPoint
from mne.io.proj import Projection
from mne.io.utils import _mult_cal_one
from mne.io.constants import FIFF

raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')


def assert_named_constants(info):
    """Assert that info['chs'] has named constants."""
    # for now we just check one
    __tracebackhide__ = True
    r = repr(info['chs'][0])
    for check in ('.*FIFFV_COORD_.*', '.*FIFFV_COIL_.*', '.*FIFF_UNIT_.*',
                  '.*FIFF_UNITM_.*',):
        assert re.match(check, r, re.DOTALL) is not None, (check, r)


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


def _test_raw_reader(reader, test_preloading=True, test_kwargs=True,
                     boundary_decimal=2, test_scaling=True, test_rank=True,
                     **kwargs):
    """Test reading, writing and slicing of raw classes.

    Parameters
    ----------
    reader : function
        Function to test.
    test_preloading : bool
        Whether not preloading is implemented for the reader. If True, both
        cases and memory mapping to file are tested.
    test_kwargs : dict
        Test _init_kwargs support.
    boundary_decimal : int
        Number of decimals up to which the boundary should match.
    **kwargs :
        Arguments for the reader. Note: Do not use preload as kwarg.
        Use ``test_preloading`` instead.

    Returns
    -------
    raw : instance of Raw
        A preloaded Raw object.
    """
    tempdir = _TempDir()
    rng = np.random.RandomState(0)
    montage = None
    if "montage" in kwargs:
        montage = kwargs['montage']
        del kwargs['montage']
    if test_preloading:
        raw = reader(preload=True, **kwargs)
        rep = repr(raw)
        assert rep.count('<') == 1
        assert rep.count('>') == 1
        if montage is not None:
            raw.set_montage(montage)
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
                assert_allclose(
                    data1, data2, err_msg='Data mismatch with preload')
                assert_allclose(times1, times2)

        # test projection vs cals and data units
        other_raw = reader(preload=False, **kwargs)
        other_raw.del_proj()
        eeg = meg = fnirs = False
        if 'eeg' in raw:
            eeg, atol = True, 1e-18
        elif 'grad' in raw:
            meg, atol = 'grad', 1e-24
        elif 'mag' in raw:
            meg, atol = 'mag', 1e-24
        elif 'hbo' in raw:
            fnirs, atol = 'hbo', 1e-10
        elif 'hbr' in raw:
            fnirs, atol = 'hbr', 1e-10
        else:
            assert 'fnirs_cw_amplitude' in raw, 'New channel type necessary?'
            fnirs, atol = 'fnirs_cw_amplitude', 1e-10
        picks = pick_types(
            other_raw.info, meg=meg, eeg=eeg, fnirs=fnirs)
        col_names = [other_raw.ch_names[pick] for pick in picks]
        proj = np.ones((1, len(picks)))
        proj /= np.sqrt(proj.shape[1])
        proj = Projection(
            data=dict(data=proj, nrow=1, row_names=None,
                      col_names=col_names, ncol=len(picks)),
            active=False)
        assert len(other_raw.info['projs']) == 0
        other_raw.add_proj(proj)
        assert len(other_raw.info['projs']) == 1
        # Orders of projector application, data loading, and reordering
        # equivalent:
        # 1. load->apply->get
        data_load_apply_get = \
            other_raw.copy().load_data().apply_proj().get_data(picks)
        # 2. apply->get (and don't allow apply->pick)
        apply = other_raw.copy().apply_proj()
        data_apply_get = apply.get_data(picks)
        data_apply_get_0 = apply.get_data(picks[0])[0]
        with pytest.raises(RuntimeError, match='loaded'):
            apply.copy().pick(picks[0]).get_data()
        # 3. apply->load->get
        data_apply_load_get = apply.copy().load_data().get_data(picks)
        data_apply_load_get_0, data_apply_load_get_1 = \
            apply.copy().load_data().pick(picks[:2]).get_data()
        # 4. reorder->apply->load->get
        all_picks = np.arange(len(other_raw.ch_names))
        reord = np.concatenate((
            picks[1::2],
            picks[0::2],
            np.setdiff1d(all_picks, picks)))
        rev = np.argsort(reord)
        assert_array_equal(reord[rev], all_picks)
        assert_array_equal(rev[reord], all_picks)
        reorder = other_raw.copy().pick(reord)
        assert reorder.ch_names == [other_raw.ch_names[r] for r in reord]
        assert reorder.ch_names[0] == other_raw.ch_names[picks[1]]
        assert_allclose(reorder.get_data([0]), other_raw.get_data(picks[1]))
        reorder_apply = reorder.copy().apply_proj()
        assert reorder_apply.ch_names == reorder.ch_names
        assert reorder_apply.ch_names[0] == apply.ch_names[picks[1]]
        assert_allclose(reorder_apply.get_data([0]), apply.get_data(picks[1]),
                        atol=1e-18)
        data_reorder_apply_load_get = \
            reorder_apply.load_data().get_data(rev[:len(picks)])
        data_reorder_apply_load_get_1 = \
            reorder_apply.copy().load_data().pick([0]).get_data()[0]
        assert reorder_apply.ch_names[0] == apply.ch_names[picks[1]]
        assert (data_load_apply_get.shape ==
                data_apply_get.shape ==
                data_apply_load_get.shape ==
                data_reorder_apply_load_get.shape)
        del apply
        # first check that our data are (probably) in the right units
        data = data_load_apply_get.copy()
        data = data - np.mean(data, axis=1, keepdims=True)  # can be offsets
        np.abs(data, out=data)
        if test_scaling:
            maxval = atol * 1e16
            assert_array_less(data, maxval)
            minval = atol * 1e6
            assert_array_less(minval, np.median(data))
        else:
            atol = 1e-7 * np.median(data)  # 1e-7 * MAD
        # ranks should all be reduced by 1
        if test_rank == 'less':
            cmp = np.less
        elif test_rank is False:
            cmp = None
        else:  # anything else is like True or 'equal'
            assert test_rank is True or test_rank == 'equal', test_rank
            cmp = np.equal
        rank_load_apply_get = np.linalg.matrix_rank(data_load_apply_get)
        rank_apply_get = np.linalg.matrix_rank(data_apply_get)
        rank_apply_load_get = np.linalg.matrix_rank(data_apply_load_get)
        if cmp is not None:
            assert cmp(rank_load_apply_get, len(col_names) - 1)
            assert cmp(rank_apply_get, len(col_names) - 1)
            assert cmp(rank_apply_load_get, len(col_names) - 1)
        # and they should all match
        t_kw = dict(
            atol=atol, err_msg='before != after, likely _mult_cal_one prob')
        assert_allclose(data_apply_get[0], data_apply_get_0, **t_kw)
        assert_allclose(data_apply_load_get_1,
                        data_reorder_apply_load_get_1, **t_kw)
        assert_allclose(data_load_apply_get[0], data_apply_load_get_0, **t_kw)
        assert_allclose(data_load_apply_get, data_apply_get, **t_kw)
        assert_allclose(data_load_apply_get, data_apply_load_get, **t_kw)
        if 'eeg' in raw:
            other_raw.del_proj()
            direct = \
                other_raw.copy().load_data().set_eeg_reference().get_data()
            other_raw.set_eeg_reference(projection=True)
            assert len(other_raw.info['projs']) == 1
            this_proj = other_raw.info['projs'][0]['data']
            assert this_proj['col_names'] == col_names
            assert this_proj['data'].shape == proj['data']['data'].shape
            assert_allclose(
                np.linalg.norm(proj['data']['data']), 1., atol=1e-6)
            assert_allclose(
                np.linalg.norm(this_proj['data']), 1., atol=1e-6)
            assert_allclose(this_proj['data'], proj['data']['data'])
            proj = other_raw.apply_proj().get_data()
            assert_allclose(proj[picks], data_load_apply_get, atol=1e-10)
            assert_allclose(proj, direct, atol=1e-10, err_msg=t_kw['err_msg'])
    else:
        raw = reader(**kwargs)
    n_samp = len(raw.times)
    assert_named_constants(raw.info)
    # smoke test for gh #9743
    ids = [id(ch['loc']) for ch in raw.info['chs']]
    assert len(set(ids)) == len(ids)

    full_data = raw._data
    assert raw.__class__.__name__ in repr(raw)  # to test repr
    assert raw.info.__class__.__name__ in repr(raw.info)
    assert isinstance(raw.info['dig'], (type(None), list))
    data_max = full_data.max()
    data_min = full_data.min()
    # these limits could be relaxed if we actually find data with
    # huge values (in SI units)
    assert data_max < 1e5
    assert data_min > -1e5
    if isinstance(raw.info['dig'], list):
        for di, d in enumerate(raw.info['dig']):
            assert isinstance(d, DigPoint), (di, d)

    # gh-5604
    meas_date = raw.info['meas_date']
    assert meas_date is None or meas_date >= _stamp_to_dt((0, 0))

    # test repr_html
    assert 'Good channels' in raw.info._repr_html_()

    # test resetting raw
    if test_kwargs:
        raw2 = reader(**raw._init_kwargs)
        assert set(raw.info.keys()) == set(raw2.info.keys())
        assert_array_equal(raw.times, raw2.times)

    # Test saving and reading
    out_fname = op.join(tempdir, 'test_raw.fif')
    raw = concatenate_raws([raw])
    raw.save(out_fname, tmax=raw.times[-1], overwrite=True, buffer_size_sec=1)

    # Test saving with not correct extension
    out_fname_h5 = op.join(tempdir, 'test_raw.h5')
    with pytest.raises(IOError, match='raw must end with .fif or .fif.gz'):
        raw.save(out_fname_h5)

    raw3 = read_raw_fif(out_fname)
    assert_named_constants(raw3.info)
    assert set(raw.info.keys()) == set(raw3.info.keys())
    assert_allclose(raw3[0:20][0], full_data[0:20], rtol=1e-6,
                    atol=1e-20)  # atol is very small but > 0
    assert_allclose(raw.times, raw3.times, atol=1e-6, rtol=1e-6)

    assert not math.isnan(raw3.info['highpass'])
    assert not math.isnan(raw3.info['lowpass'])
    assert not math.isnan(raw.info['highpass'])
    assert not math.isnan(raw.info['lowpass'])

    assert raw3.info['kit_system_id'] == raw.info['kit_system_id']

    # Make sure concatenation works
    first_samp = raw.first_samp
    last_samp = raw.last_samp
    concat_raw = concatenate_raws([raw.copy(), raw])
    assert concat_raw.n_times == 2 * raw.n_times
    assert concat_raw.first_samp == first_samp
    assert concat_raw.last_samp - last_samp + first_samp == last_samp + 1
    idx = np.where(concat_raw.annotations.description == 'BAD boundary')[0]

    expected_bad_boundary_onset = raw._last_time

    assert_array_almost_equal(concat_raw.annotations.onset[idx],
                              expected_bad_boundary_onset,
                              decimal=boundary_decimal)

    if raw.info['meas_id'] is not None:
        for key in ['secs', 'usecs', 'version']:
            assert raw.info['meas_id'][key] == raw3.info['meas_id'][key]
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

    # Test picking with and without preload
    if test_preloading:
        preload_kwargs = (dict(preload=True), dict(preload=False))
    else:
        preload_kwargs = (dict(),)
    n_ch = len(raw.ch_names)
    picks = rng.permutation(n_ch)
    for preload_kwarg in preload_kwargs:
        these_kwargs = kwargs.copy()
        these_kwargs.update(preload_kwarg)
        # don't use the same filename or it could create problems
        if isinstance(these_kwargs.get('preload', None), str) and \
                op.isfile(these_kwargs['preload']):
            these_kwargs['preload'] += '-1'
        whole_raw = reader(**these_kwargs)
        print(whole_raw)  # __repr__
        assert n_ch >= 2
        picks_1 = picks[:n_ch // 2]
        picks_2 = picks[n_ch // 2:]
        raw_1 = whole_raw.copy().pick(picks_1)
        raw_2 = whole_raw.copy().pick(picks_2)
        data, times = whole_raw[:]
        data_1, times_1 = raw_1[:]
        data_2, times_2 = raw_2[:]
        assert_array_equal(times, times_1)
        assert_array_equal(data[picks_1], data_1)
        assert_array_equal(times, times_2,)
        assert_array_equal(data[picks_2], data_2)

    # Make sure that writing info to h5 format
    # (all fields should be compatible)
    if check_version('h5io'):
        read_hdf5, write_hdf5 = _import_h5io_funcs()
        fname_h5 = op.join(tempdir, 'info.h5')
        with _writing_info_hdf5(raw.info):
            write_hdf5(fname_h5, raw.info)
        new_info = Info(read_hdf5(fname_h5))
        assert object_diff(new_info, raw.info) == ''

    # Make sure that changing directory does not break anything
    if test_preloading:
        these_kwargs = kwargs.copy()
        key = None
        for key in ('fname',
                    'input_fname',  # artemis123
                    'vhdr_fname',  # BV
                    'pdf_fname',  # BTi
                    'directory',  # CTF
                    'filename',  # nedf
                    ):
            try:
                fname = kwargs[key]
            except KeyError:
                key = None
            else:
                break
        # len(kwargs) == 0 for the fake arange reader
        if len(kwargs):
            assert key is not None, sorted(kwargs.keys())
            this_fname = fname[0] if isinstance(fname, list) else fname
            dirname = op.dirname(this_fname)
            these_kwargs[key] = op.basename(this_fname)
            these_kwargs['preload'] = False
            orig_dir = os.getcwd()
            try:
                os.chdir(dirname)
                raw_chdir = reader(**these_kwargs)
            finally:
                os.chdir(orig_dir)
            raw_chdir.load_data()

    # make sure that cropping works (with first_samp shift)
    if n_samp >= 50:  # we crop to this number of samples below
        for t_prop in (0., 0.5):
            _test_raw_crop(reader, t_prop, kwargs)
            if test_preloading:
                use_kwargs = kwargs.copy()
                use_kwargs['preload'] = True
                _test_raw_crop(reader, t_prop, use_kwargs)

    # make sure electrode-like sensor locations show up as dig points
    eeg_dig = [d for d in (raw.info['dig'] or [])
               if d['kind'] == _dig_kind_dict['eeg']]
    pick_kwargs = dict()
    for t in _ELECTRODE_CH_TYPES + ('fnirs',):
        pick_kwargs[t] = True
    dig_picks = pick_types(raw.info, exclude=(), **pick_kwargs)
    dig_types = _ELECTRODE_CH_TYPES + _FNIRS_CH_TYPES_SPLIT
    assert (len(dig_picks) > 0) == any(t in raw for t in dig_types)
    if len(dig_picks):
        eeg_loc = np.array([  # eeg_loc a bit of a misnomer to match eeg_dig
            raw.info['chs'][pick]['loc'][:3] for pick in dig_picks])
        eeg_loc = eeg_loc[np.isfinite(eeg_loc).all(axis=1)]
        if len(eeg_loc):
            if 'fnirs_cw_amplitude' in raw:
                assert 2 * len(eeg_dig) >= len(eeg_loc)
            else:
                assert len(eeg_dig) >= len(eeg_loc)  # could have some excluded
    # make sure that dig points in head coords implies that fiducials are
    # present
    if len(raw.info['dig'] or []) > 0:
        card_pts = [d for d in raw.info['dig']
                    if d['kind'] == _dig_kind_dict['cardinal']]
        eeg_dig_head = [
            d for d in eeg_dig if d['coord_frame'] == FIFF.FIFFV_COORD_HEAD]
        if len(eeg_dig_head):
            assert len(card_pts) == 3, 'Cardinal points missing'
        if len(card_pts) == 3:  # they should all be in head coords then
            assert len(eeg_dig_head) == len(eeg_dig)

    return raw


def _test_raw_crop(reader, t_prop, kwargs):
    raw_1 = reader(**kwargs)
    n_samp = 50  # crop to this number of samples (per instance)
    crop_t = n_samp / raw_1.info['sfreq']
    t_start = t_prop * crop_t  # also crop to some fraction into the first inst
    extra = f' t_start={t_start}, preload={kwargs.get("preload", False)}'
    stop = (n_samp - 1) / raw_1.info['sfreq']
    raw_1.crop(0, stop)
    assert len(raw_1.times) == 50
    first_time = raw_1.first_time
    atol = 0.5 / raw_1.info['sfreq']
    assert_allclose(raw_1.times[-1], stop, atol=atol)
    raw_2, raw_3 = raw_1.copy(), raw_1.copy()
    t_tot = raw_1.times[-1] * 3 + 2. / raw_1.info['sfreq']
    raw_concat = concatenate_raws([raw_1, raw_2, raw_3])
    assert len(raw_concat._filenames) == 3
    assert_allclose(raw_concat.times[-1], t_tot)
    assert_allclose(raw_concat.first_time, first_time)
    # keep all instances, but crop to t_start at the beginning
    raw_concat.crop(t_start, None)
    assert len(raw_concat._filenames) == 3
    assert_allclose(raw_concat.times[-1], t_tot - t_start, atol=atol)
    assert_allclose(
        raw_concat.first_time, first_time + t_start, atol=atol,
        err_msg=f'Base concat, {extra}')
    # drop the first instance
    raw_concat.crop(crop_t, None)
    assert len(raw_concat._filenames) == 2
    assert_allclose(
        raw_concat.times[-1], t_tot - t_start - crop_t, atol=atol)
    assert_allclose(
        raw_concat.first_time, first_time + t_start + crop_t,
        atol=atol, err_msg=f'Dropping one, {extra}')
    # drop the second instance, leaving just one
    raw_concat.crop(crop_t, None)
    assert len(raw_concat._filenames) == 1
    assert_allclose(
        raw_concat.times[-1], t_tot - t_start - 2 * crop_t, atol=atol)
    assert_allclose(
        raw_concat.first_time, first_time + t_start + 2 * crop_t,
        atol=atol, err_msg=f'Dropping two, {extra}')


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
    raw = read_raw_fif(raw_fname)

    # Test original (non-rounding) indexing behavior
    orig_inds = raw.time_as_index(raw.times)
    assert len(set(orig_inds)) != len(orig_inds)

    # Test new (rounding) indexing behavior
    new_inds = raw.time_as_index(raw.times, use_rounding=True)
    assert_array_equal(new_inds, np.arange(len(raw.times)))


@pytest.mark.parametrize('meas_date', [None, "orig"])
@pytest.mark.parametrize('first_samp', [0, 10000])
def test_crop_by_annotations(meas_date, first_samp):
    """Test crop by annotations of raw."""
    raw = read_raw_fif(raw_fname)

    if meas_date is None:
        raw.set_meas_date(None)

    raw = mne.io.RawArray(raw.get_data(), raw.info, first_samp=first_samp)

    onset = np.array([0, 1.5], float)
    if meas_date is not None:
        onset += raw.first_time
    annot = mne.Annotations(
        onset=onset,
        duration=[1, 0.5],
        description=["a", "b"],
        orig_time=raw.info['meas_date'])

    raw.set_annotations(annot)
    raws = raw.crop_by_annotations()
    assert len(raws) == 2
    assert len(raws[0].annotations) == 1
    assert raws[0].times[-1] == pytest.approx(annot[:1].duration[0], rel=1e-3)
    assert raws[0].annotations.description[0] == annot.description[0]
    assert len(raws[1].annotations) == 1
    assert raws[1].times[-1] == pytest.approx(annot[1:2].duration[0], rel=5e-3)
    assert raws[1].annotations.description[0] == annot.description[1]


@pytest.mark.parametrize('offset, origin', [
    pytest.param(0, None, id='times in s. relative to first_samp (default)'),
    pytest.param(0, 2.0, id='times in s. relative to first_samp'),
    pytest.param(1, 1.0, id='times in s. relative to meas_date'),
    pytest.param(2, 0.0, id='absolute times in s. relative to 0')])
def test_time_as_index_ref(offset, origin):
    """Test indexing of raw times."""
    info = create_info(ch_names=10, sfreq=10.)
    raw = RawArray(data=np.empty((10, 10)), info=info, first_samp=10)
    raw.set_meas_date(1)

    relative_times = raw.times
    inds = raw.time_as_index(relative_times + offset,
                             use_rounding=True,
                             origin=origin)
    assert_array_equal(inds, np.arange(raw.n_times))


def test_meas_date_orig_time():
    """Test the relation between meas_time in orig_time."""
    # meas_time is set and orig_time is set:
    # clips the annotations based on raw.data and resets the annotation based
    # on raw.info['meas_date]
    raw = _raw_annot(1, 1.5)
    assert raw.annotations.orig_time == _stamp_to_dt((1, 0))
    assert raw.annotations.onset[0] == 1

    # meas_time is set and orig_time is None:
    # Consider annot.orig_time to be raw.frist_sample, clip and reset
    # annotations to have the raw.annotations.orig_time == raw.info['meas_date]
    raw = _raw_annot(1, None)
    assert raw.annotations.orig_time == _stamp_to_dt((1, 0))
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
    assert raw.annotations.onset[0] == 1.5
    assert raw.annotations.duration[0] == 0.2


def test_get_data_reject():
    """Test if reject_by_annotation is working correctly."""
    fs = 256
    ch_names = ["C3", "Cz", "C4"]
    info = create_info(ch_names, sfreq=fs)
    raw = RawArray(np.zeros((len(ch_names), 10 * fs)), info)
    raw.set_annotations(Annotations(onset=[2, 4], duration=[3, 2],
                                    description="bad"))

    with catch_logging() as log:
        data = raw.get_data(reject_by_annotation="omit", verbose=True)
        msg = ('Omitting 1024 of 2560 (40.00%) samples, retaining 1536' +
               ' (60.00%) samples.')
        assert log.getvalue().strip() == msg
    assert data.shape == (len(ch_names), 1536)
    with catch_logging() as log:
        data = raw.get_data(reject_by_annotation="nan", verbose=True)
        msg = ('Setting 1024 of 2560 (40.00%) samples to NaN, retaining 1536' +
               ' (60.00%) samples.')
        assert log.getvalue().strip() == msg
    assert data.shape == (len(ch_names), 2560)  # shape doesn't change
    assert np.isnan(data).sum() == 3072  # but NaNs are introduced instead


def test_5839():
    """Test concatenating raw objects with annotations."""
    # Global Time 0         1         2         3         4
    #             .
    #      raw_A  |---------XXXXXXXXXX
    #      annot  |--------------AA
    #    latency  .         0    0    1    1    2    2    3
    #             .              5    0    5    0    5    0
    #
    #      raw_B  .                   |---------YYYYYYYYYY
    #      annot  .                   |--------------AA
    #    latency  .                             0         1
    #             .                                  5    0
    #             .
    #     output  |---------XXXXXXXXXXYYYYYYYYYY
    #      annot  |--------------AA---|----AA
    #    latency  .         0    0    1    1    2    2    3
    #             .              5    0    5    0    5    0
    #
    EXPECTED_ONSET = [1.5, 2., 2., 2.5]
    EXPECTED_DURATION = [0.2, 0., 0., 0.2]
    EXPECTED_DESCRIPTION = ['dummy', 'BAD boundary', 'EDGE boundary', 'dummy']

    def raw_factory(meas_date):
        raw = RawArray(data=np.empty((10, 10)),
                       info=create_info(ch_names=10, sfreq=10.),
                       first_samp=10)
        raw.set_meas_date(meas_date)
        raw.set_annotations(annotations=Annotations(onset=[.5],
                                                    duration=[.2],
                                                    description='dummy',
                                                    orig_time=None))
        return raw

    raw_A, raw_B = [raw_factory((x, 0)) for x in [0, 2]]
    raw_A.append(raw_B)

    assert_array_equal(raw_A.annotations.onset, EXPECTED_ONSET)
    assert_array_equal(raw_A.annotations.duration, EXPECTED_DURATION)
    assert_array_equal(raw_A.annotations.description, EXPECTED_DESCRIPTION)
    assert raw_A.annotations.orig_time == _stamp_to_dt((0, 0))


def test_repr():
    """Test repr of Raw."""
    sfreq = 256
    info = create_info(3, sfreq)
    raw = RawArray(np.zeros((3, 10 * sfreq)), info)
    r = repr(raw)
    assert re.search('<RawArray | 3 x 2560 (10.0 s), ~.* kB, data loaded>',
                     r) is not None, r
    assert raw._repr_html_()


# A class that sets channel data to np.arange, for testing _test_raw_reader
class _RawArange(BaseRaw):

    def __init__(self, preload=False, verbose=None):
        info = create_info(list(str(x) for x in range(1, 9)), 1000., 'eeg')
        super().__init__(info, preload, last_samps=(999,), verbose=verbose)
        assert len(self.times) == 1000

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        one = np.full((8, stop - start), np.nan)
        one[idx] = np.arange(1, 9)[idx, np.newaxis]
        _mult_cal_one(data, one, idx, cals, mult)


def _read_raw_arange(preload=False, verbose=None):
    return _RawArange(preload, verbose)


def test_test_raw_reader():
    """Test _test_raw_reader."""
    _test_raw_reader(_read_raw_arange, test_scaling=False, test_rank='less')


@pytest.mark.slowtest
def test_describe_print():
    """Test print output of describe method."""
    fname = Path(__file__).parent / "data" / "test_raw.fif"
    raw = read_raw_fif(fname)

    # test print output
    f = StringIO()
    with redirect_stdout(f):
        raw.describe()
    s = f.getvalue().strip().split("\n")
    assert len(s) == 378
    # Can be 3.1, 3.3, etc.
    assert re.match(
        r'<Raw | test_raw.fif, 376 x 14400 (24\.0 s), '
        r'~3\.. MB, data not loaded>', s[0]) is not None, s[0]
    assert s[1] == " ch  name      type  unit         min         Q1     median         Q3        max"  # noqa
    assert s[2] == "  0  MEG 0113  GRAD  fT/cm    -221.80     -38.57      -9.64      19.29     414.67"  # noqa
    assert s[-1] == "375  EOG 061   EOG   µV       -231.41     271.28     277.16     285.66     334.69"  # noqa


@requires_pandas
@pytest.mark.slowtest
def test_describe_df():
    """Test returned data frame of describe method."""
    fname = Path(__file__).parent / "data" / "test_raw.fif"
    raw = read_raw_fif(fname)

    df = raw.describe(data_frame=True)
    assert df.shape == (376, 8)
    assert (df.columns.tolist() == ["name", "type", "unit", "min", "Q1",
                                    "median", "Q3", "max"])
    assert df.index.name == "ch"
    assert_allclose(df.iloc[0, 3:].astype(float),
                    np.array([-2.218017605790535e-11,
                              -3.857421923113974e-12,
                              -9.643554807784935e-13,
                              1.928710961556987e-12,
                              4.146728567347522e-11]))


def test_get_data_units():
    """Test the "units" argument of get_data method."""
    # Test the unit conversion function
    assert _get_scaling('eeg', 'uV') == 1e6
    assert _get_scaling('eeg', 'dV') == 1e1
    assert _get_scaling('eeg', 'pV') == 1e12
    assert _get_scaling('mag', 'fT') == 1e15
    assert _get_scaling('grad', 'T/m') == 1
    assert _get_scaling('grad', 'T/mm') == 1e-3
    assert _get_scaling('grad', 'fT/m') == 1e15
    assert _get_scaling('grad', 'fT/cm') == 1e13
    assert _get_scaling('csd', 'uV/cm²') == 1e2

    fname = Path(__file__).parent / "data" / "test_raw.fif"
    raw = read_raw_fif(fname)

    last = np.array([4.63803098e-05, 7.66563736e-05, 2.71933595e-04])
    last_eeg = np.array([7.12207023e-05, 4.63803098e-05, 7.66563736e-05])
    last_grad = np.array([-3.85742192e-12,  9.64355481e-13, -1.06079103e-11])

    # None
    data_none = raw.get_data()
    assert data_none.shape == (376, 14400)
    assert_array_almost_equal(data_none[-3:, -1], last)

    # str: unit no conversion
    data_str_noconv = raw.get_data(picks=['eeg'], units='V')
    assert data_str_noconv.shape == (60, 14400)
    assert_array_almost_equal(data_str_noconv[-3:, -1], last_eeg)
    # str: simple unit
    data_str_simple = raw.get_data(picks=['eeg'], units='uV')
    assert data_str_simple.shape == (60, 14400)
    assert_array_almost_equal(data_str_simple[-3:, -1], last_eeg * 1e6)
    # str: fraction unit
    data_str_fraction = raw.get_data(picks=['grad'], units='fT/cm')
    assert data_str_fraction.shape == (204, 14400)
    assert_array_almost_equal(data_str_fraction[-3:, -1],
                              last_grad * (1e15 / 1e2))
    # str: more than one channel type but one with unit
    data_str_simplestim = raw.get_data(picks=['eeg', 'stim'], units='V')
    assert data_str_simplestim.shape == (69, 14400)
    assert_array_almost_equal(data_str_simplestim[-3:, -1], last_eeg)
    # str: too many channels
    with pytest.raises(ValueError, match='more than one channel'):
        raw.get_data(units='uV')
    # str: invalid unit
    with pytest.raises(ValueError, match='is not a valid unit'):
        raw.get_data(picks=['eeg'], units='fV/cm')

    # dict: combination of simple and fraction units
    data_dict = raw.get_data(units=dict(grad='fT/cm', mag='fT', eeg='uV'))
    assert data_dict.shape == (376, 14400)
    assert_array_almost_equal(data_dict[0, -1],
                              -3.857421923113974e-12 * (1e15 / 1e2))
    assert_array_almost_equal(data_dict[2, -1], -2.1478272253525944e-13 * 1e15)
    assert_array_almost_equal(data_dict[-2, -1], 7.665637356879529e-05 * 1e6)
    # dict: channel type not in instance
    data_dict_notin = raw.get_data(units=dict(hbo='uM'))
    assert data_dict_notin.shape == (376, 14400)
    assert_array_almost_equal(data_dict_notin[-3:, -1], last)
    # dict: one invalid unit
    with pytest.raises(ValueError, match='is not a valid unit'):
        raw.get_data(units=dict(grad='fT/cV', mag='fT', eeg='uV'))
    # dict: one invalid channel type
    with pytest.raises(KeyError, match='is not a channel type'):
        raw.get_data(units=dict(bad_type='fT/cV', mag='fT', eeg='uV'))

    # not the good type
    with pytest.raises(TypeError, match='instance of None, str, or dict'):
        raw.get_data(units=['fT/cm', 'fT', 'uV'])


def test_repr_dig_point():
    """Test printing of DigPoint."""
    dp = DigPoint(r=np.arange(3), coord_frame=FIFF.FIFFV_COORD_HEAD,
                  kind=FIFF.FIFFV_POINT_EEG, ident=0)
    assert 'mm' in repr(dp)

    dp = DigPoint(r=np.arange(3), coord_frame=FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
                  kind=FIFF.FIFFV_POINT_CARDINAL, ident=0)
    assert 'mm' not in repr(dp)
    assert 'voxel' in repr(dp)


def test_get_data_tmin_tmax():
    """Test tmin and tmax parameters of get_data method."""
    fname = Path(__file__).parent / "data" / "test_raw.fif"
    raw = read_raw_fif(fname)

    # tmin and tmax just use time_as_index under the hood
    tmin, tmax = (1, 9)
    d1 = raw.get_data()
    d2 = raw.get_data(tmin=tmin, tmax=tmax)

    idxs = raw.time_as_index([tmin, tmax])
    assert_allclose(d1[:, idxs[0]:idxs[1]], d2)

    # specifying a too low tmin truncates to idx 0
    d3 = raw.get_data(tmin=-5)
    assert_allclose(d3, d1)

    # specifying a too high tmax truncates to idx n_times
    d4 = raw.get_data(tmax=1e6)
    assert_allclose(d4, d1)

    # when start/stop are passed, tmin/tmax are ignored
    d5 = raw.get_data(start=1, stop=2, tmin=tmin, tmax=tmax)
    assert d5.shape[1] == 1

    # validate inputs are properly raised
    with pytest.raises(TypeError, match='start must be .* int'):
        raw.get_data(start=None)

    with pytest.raises(TypeError, match='stop must be .* int'):
        raw.get_data(stop=2.3)

    with pytest.raises(TypeError, match='tmin must be .* float'):
        raw.get_data(tmin=[1, 2])

    with pytest.raises(TypeError, match='tmax must be .* float'):
        raw.get_data(tmax=[1, 2])
