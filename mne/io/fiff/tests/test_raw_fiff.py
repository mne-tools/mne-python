# -*- coding: utf-8 -*-
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from functools import partial
from io import BytesIO
import os.path as op
import pathlib
import pickle
import sys

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
import pytest

from mne.datasets import testing
from mne.filter import filter_data
from mne.io.constants import FIFF
from mne.io import RawArray, concatenate_raws, read_raw_fif
from mne.io.tests.test_raw import _test_concat, _test_raw_reader
from mne import (concatenate_events, find_events, equalize_channels,
                 compute_proj_raw, pick_types, pick_channels, create_info,
                 pick_info)
from mne.utils import (requires_pandas, assert_object_equal, _dt_to_stamp,
                       requires_mne, run_subprocess, run_tests_if_main,
                       assert_and_remove_boundary_annot)
from mne.annotations import Annotations

testing_path = testing.data_path(download=False)
data_dir = op.join(testing_path, 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')
ms_fname = op.join(testing_path, 'SSS', 'test_move_anon_raw.fif')
skip_fname = op.join(testing_path, 'misc', 'intervalrecording_raw.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
test_fif_gz_fname = op.join(base_dir, 'test_raw.fif.gz')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')
fif_bad_marked_fname = op.join(base_dir, 'test_withbads_raw.fif')
bad_file_works = op.join(base_dir, 'test_bads.txt')
bad_file_wrong = op.join(base_dir, 'test_wrong_bads.txt')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


@testing.requires_testing_data
def test_acq_skip(tmpdir):
    """Test treatment of acquisition skips."""
    raw = read_raw_fif(skip_fname, preload=True)
    picks = [1, 2, 10]
    assert len(raw.times) == 17000
    annotations = raw.annotations
    assert len(annotations) == 3  # there are 3 skips
    assert_allclose(annotations.onset, [14, 19, 23])
    assert_allclose(annotations.duration, [2., 2., 3.])  # inclusive!
    data, times = raw.get_data(
        picks, reject_by_annotation='omit', return_times=True)
    expected_data, expected_times = zip(raw[picks, :2000],
                                        raw[picks, 4000:7000],
                                        raw[picks, 9000:11000],
                                        raw[picks, 14000:17000])
    expected_times = np.concatenate(list(expected_times), axis=-1)
    assert_allclose(times, expected_times)
    expected_data = list(expected_data)
    assert_allclose(data, np.concatenate(expected_data, axis=-1), atol=1e-22)

    # Check that acquisition skips are handled properly in filtering
    kwargs = dict(l_freq=None, h_freq=50., fir_design='firwin')
    raw_filt = raw.copy().filter(picks=picks, **kwargs)
    for data in expected_data:
        filter_data(data, raw.info['sfreq'], copy=False, **kwargs)
    data = raw_filt.get_data(picks, reject_by_annotation='omit')
    assert_allclose(data, np.concatenate(expected_data, axis=-1), atol=1e-22)

    # Check that acquisition skips are handled properly during I/O
    fname = tmpdir.join('test_raw.fif')
    raw.save(fname, fmt=raw.orig_format)
    # first: file size should not increase much (orig data is missing
    # 7 of 17 buffers, so if we write them out it should increase the file
    # size quite a bit.
    orig_size = op.getsize(skip_fname)
    new_size = op.getsize(fname)
    max_size = int(1.05 * orig_size)  # almost the same + annotations
    assert new_size < max_size, (new_size, max_size)
    raw_read = read_raw_fif(fname)
    assert raw_read.annotations is not None
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw_read[:][0], raw[:][0], atol=1e-17)
    # Saving with a bad buffer length emits warning
    raw.pick_channels(raw.ch_names[:2])
    with pytest.warns(None) as w:
        raw.save(fname, buffer_size_sec=0.5, overwrite=True)
    assert len(w) == 0
    with pytest.warns(RuntimeWarning, match='did not fit evenly'):
        raw.save(fname, buffer_size_sec=2., overwrite=True)


def test_fix_types():
    """Test fixing of channel types."""
    for fname, change in ((hp_fif_fname, True), (test_fif_fname, False),
                          (ctf_fname, False)):
        raw = read_raw_fif(fname)
        mag_picks = pick_types(raw.info, meg='mag')
        other_picks = np.setdiff1d(np.arange(len(raw.ch_names)), mag_picks)
        # we don't actually have any files suffering from this problem, so
        # fake it
        if change:
            for ii in mag_picks:
                raw.info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T2
        orig_types = np.array([ch['coil_type'] for ch in raw.info['chs']])
        raw.fix_mag_coil_types()
        new_types = np.array([ch['coil_type'] for ch in raw.info['chs']])
        if not change:
            assert_array_equal(orig_types, new_types)
        else:
            assert_array_equal(orig_types[other_picks], new_types[other_picks])
            assert ((orig_types[mag_picks] != new_types[mag_picks]).all())
            assert ((new_types[mag_picks] ==
                     FIFF.FIFFV_COIL_VV_MAG_T3).all())


def test_concat(tmpdir):
    """Test RawFIF concatenation."""
    # we trim the file to save lots of memory and some time
    raw = read_raw_fif(test_fif_fname)
    raw.crop(0, 2.)
    test_name = tmpdir.join('test_raw.fif')
    raw.save(test_name)
    # now run the standard test
    _test_concat(partial(read_raw_fif), test_name)


@testing.requires_testing_data
def test_hash_raw():
    """Test hashing raw objects."""
    raw = read_raw_fif(fif_fname)
    pytest.raises(RuntimeError, raw.__hash__)
    raw = read_raw_fif(fif_fname).crop(0, 0.5)
    raw_size = raw._size
    raw.load_data()
    raw_load_size = raw._size
    assert (raw_size < raw_load_size)
    raw_2 = read_raw_fif(fif_fname).crop(0, 0.5)
    raw_2.load_data()
    assert hash(raw) == hash(raw_2)
    # do NOT use assert_equal here, failing output is terrible
    assert pickle.dumps(raw) == pickle.dumps(raw_2)

    raw_2._data[0, 0] -= 1
    assert hash(raw) != hash(raw_2)


@testing.requires_testing_data
def test_maxshield():
    """Test maxshield warning."""
    with pytest.warns(RuntimeWarning, match='Internal Active Shielding') as w:
        read_raw_fif(ms_fname, allow_maxshield=True)
    assert ('test_raw_fiff.py' in w[0].filename)


@testing.requires_testing_data
def test_subject_info(tmpdir):
    """Test reading subject information."""
    raw = read_raw_fif(fif_fname).crop(0, 1)
    assert (raw.info['subject_info'] is None)
    # fake some subject data
    keys = ['id', 'his_id', 'last_name', 'first_name', 'birthday', 'sex',
            'hand']
    vals = [1, 'foobar', 'bar', 'foo', (1901, 2, 3), 0, 1]
    subject_info = dict()
    for key, val in zip(keys, vals):
        subject_info[key] = val
    raw.info['subject_info'] = subject_info
    out_fname = tmpdir.join('test_subj_info_raw.fif')
    raw.save(out_fname, overwrite=True)
    raw_read = read_raw_fif(out_fname)
    for key in keys:
        assert subject_info[key] == raw_read.info['subject_info'][key]
    assert raw.info['meas_date'] == raw_read.info['meas_date']

    for key in ['secs', 'usecs', 'version']:
        assert raw.info['meas_id'][key] == raw_read.info['meas_id'][key]
    assert_array_equal(raw.info['meas_id']['machid'],
                       raw_read.info['meas_id']['machid'])


@testing.requires_testing_data
def test_copy_append():
    """Test raw copying and appending combinations."""
    raw = read_raw_fif(fif_fname, preload=True).copy()
    raw_full = read_raw_fif(fif_fname)
    raw_full.append(raw)
    data = raw_full[:, :][0]
    assert data.shape[1] == 2 * raw._data.shape[1]


@testing.requires_testing_data
def test_output_formats(tmpdir):
    """Test saving and loading raw data using multiple formats."""
    formats = ['short', 'int', 'single', 'double']
    tols = [1e-4, 1e-7, 1e-7, 1e-15]

    # let's fake a raw file with different formats
    raw = read_raw_fif(test_fif_fname).crop(0, 1)

    temp_file = tmpdir.join('raw.fif')
    for ii, (fmt, tol) in enumerate(zip(formats, tols)):
        # Let's test the overwriting error throwing while we're at it
        if ii > 0:
            pytest.raises(IOError, raw.save, temp_file, fmt=fmt)
        raw.save(temp_file, fmt=fmt, overwrite=True)
        raw2 = read_raw_fif(temp_file)
        raw2_data = raw2[:, :][0]
        assert_allclose(raw2_data, raw[:, :][0], rtol=tol, atol=1e-25)
        assert raw2.orig_format == fmt


def _compare_combo(raw, new, times, n_times):
    """Compare data."""
    for ti in times:  # let's do a subset of points for speed
        orig = raw[:, ti % n_times][0]
        # these are almost_equals because of possible dtype differences
        assert_allclose(orig, new[:, ti][0])


@pytest.mark.slowtest
@testing.requires_testing_data
def test_multiple_files(tmpdir):
    """Test loading multiple files simultaneously."""
    # split file
    raw = read_raw_fif(fif_fname).crop(0, 10)
    raw.load_data()
    raw.load_data()  # test no operation
    split_size = 3.  # in seconds
    sfreq = raw.info['sfreq']
    nsamp = (raw.last_samp - raw.first_samp)
    tmins = np.round(np.arange(0., nsamp, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp]))
    tmaxs /= sfreq
    tmins /= sfreq
    assert raw.n_times == len(raw.times)

    # going in reverse order so the last fname is the first file (need later)
    raws = [None] * len(tmins)
    for ri in range(len(tmins) - 1, -1, -1):
        fname = tmpdir.join('test_raw_split-%d_raw.fif' % ri)
        raw.save(fname, tmin=tmins[ri], tmax=tmaxs[ri])
        raws[ri] = read_raw_fif(fname)
        assert (len(raws[ri].times) ==
                int(round((tmaxs[ri] - tmins[ri]) *
                          raw.info['sfreq'])) + 1)  # + 1 b/c inclusive
    events = [find_events(r, stim_channel='STI 014') for r in raws]
    last_samps = [r.last_samp for r in raws]
    first_samps = [r.first_samp for r in raws]

    # test concatenation of split file
    pytest.raises(ValueError, concatenate_raws, raws, True, events[1:])
    all_raw_1, events1 = concatenate_raws(raws, preload=False,
                                          events_list=events)
    assert_allclose(all_raw_1.times, raw.times)
    assert raw.first_samp == all_raw_1.first_samp
    assert raw.last_samp == all_raw_1.last_samp
    assert_allclose(raw[:, :][0], all_raw_1[:, :][0])
    raws[0] = read_raw_fif(fname)
    all_raw_2 = concatenate_raws(raws, preload=True)
    assert_allclose(raw[:, :][0], all_raw_2[:, :][0])

    # test proper event treatment for split files
    events2 = concatenate_events(events, first_samps, last_samps)
    events3 = find_events(all_raw_2, stim_channel='STI 014')
    assert_array_equal(events1, events2)
    assert_array_equal(events1, events3)

    # test various methods of combining files
    raw = read_raw_fif(fif_fname, preload=True)
    n_times = raw.n_times
    # make sure that all our data match
    times = list(range(0, 2 * n_times, 999))
    # add potentially problematic points
    times.extend([n_times - 1, n_times, 2 * n_times - 1])

    raw_combo0 = concatenate_raws([read_raw_fif(f)
                                   for f in [fif_fname, fif_fname]],
                                  preload=True)
    _compare_combo(raw, raw_combo0, times, n_times)
    raw_combo = concatenate_raws([read_raw_fif(f)
                                  for f in [fif_fname, fif_fname]],
                                 preload=False)
    _compare_combo(raw, raw_combo, times, n_times)
    raw_combo = concatenate_raws([read_raw_fif(f)
                                  for f in [fif_fname, fif_fname]],
                                 preload='memmap8.dat')
    _compare_combo(raw, raw_combo, times, n_times)
    assert raw[:, :][0].shape[1] * 2 == raw_combo0[:, :][0].shape[1]
    assert raw_combo0[:, :][0].shape[1] == raw_combo0.n_times

    # with all data preloaded, result should be preloaded
    raw_combo = read_raw_fif(fif_fname, preload=True)
    raw_combo.append(read_raw_fif(fif_fname, preload=True))
    assert (raw_combo.preload is True)
    assert raw_combo.n_times == raw_combo._data.shape[1]
    _compare_combo(raw, raw_combo, times, n_times)

    # with any data not preloaded, don't set result as preloaded
    raw_combo = concatenate_raws([read_raw_fif(fif_fname, preload=True),
                                  read_raw_fif(fif_fname, preload=False)])
    assert (raw_combo.preload is False)
    assert_array_equal(find_events(raw_combo, stim_channel='STI 014'),
                       find_events(raw_combo0, stim_channel='STI 014'))
    _compare_combo(raw, raw_combo, times, n_times)

    # user should be able to force data to be preloaded upon concat
    raw_combo = concatenate_raws([read_raw_fif(fif_fname, preload=False),
                                  read_raw_fif(fif_fname, preload=True)],
                                 preload=True)
    assert (raw_combo.preload is True)
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([read_raw_fif(fif_fname, preload=False),
                                  read_raw_fif(fif_fname, preload=True)],
                                 preload='memmap3.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([
        read_raw_fif(fif_fname, preload=True),
        read_raw_fif(fif_fname, preload=True)], preload='memmap4.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([
        read_raw_fif(fif_fname, preload=False),
        read_raw_fif(fif_fname, preload=False)], preload='memmap5.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    # verify that combining raws with different projectors throws an exception
    raw.add_proj([], remove_existing=True)
    pytest.raises(ValueError, raw.append,
                  read_raw_fif(fif_fname, preload=True))

    # now test event treatment for concatenated raw files
    events = [find_events(raw, stim_channel='STI 014'),
              find_events(raw, stim_channel='STI 014')]
    last_samps = [raw.last_samp, raw.last_samp]
    first_samps = [raw.first_samp, raw.first_samp]
    events = concatenate_events(events, first_samps, last_samps)
    events2 = find_events(raw_combo0, stim_channel='STI 014')
    assert_array_equal(events, events2)

    # check out the len method
    assert len(raw) == raw.n_times
    assert len(raw) == raw.last_samp - raw.first_samp + 1


@testing.requires_testing_data
def test_split_files(tmpdir):
    """Test writing and reading of split raw files."""
    raw_1 = read_raw_fif(fif_fname, preload=True)
    # Test a very close corner case
    raw_crop = raw_1.copy().crop(0, 1.)

    assert_allclose(raw_1.buffer_size_sec, 10., atol=1e-2)  # samp rate
    split_fname = tmpdir.join('split_raw_meg.fif')
    # intended filenames
    split_fname_elekta_part2 = tmpdir.join('split_raw_meg-1.fif')
    split_fname_bids_part1 = tmpdir.join('split_raw_split-01_meg.fif')
    split_fname_bids_part2 = tmpdir.join('split_raw_split-02_meg.fif')
    raw_1.set_annotations(Annotations([2.], [5.5], 'test'))
    raw_1.save(split_fname, buffer_size_sec=1.0, split_size='10MB')

    # check that the filenames match the intended pattern
    assert op.exists(split_fname_elekta_part2)
    # check that filenames are being formatted correctly for BIDS
    raw_1.save(split_fname, buffer_size_sec=1.0, split_size='10MB',
               split_naming='bids', overwrite=True)
    assert op.exists(split_fname_bids_part1)
    assert op.exists(split_fname_bids_part2)

    annot = Annotations(np.arange(20), np.ones((20,)), 'test')
    raw_1.set_annotations(annot)
    split_fname = op.join(tmpdir, 'split_raw.fif')
    raw_1.save(split_fname, buffer_size_sec=1.0, split_size='10MB')
    raw_2 = read_raw_fif(split_fname)
    assert_allclose(raw_2.buffer_size_sec, 1., atol=1e-2)  # samp rate
    assert_allclose(raw_1.annotations.onset, raw_2.annotations.onset)
    assert_allclose(raw_1.annotations.duration, raw_2.annotations.duration,
                    rtol=0.001 / raw_2.info['sfreq'])
    assert_array_equal(raw_1.annotations.description,
                       raw_2.annotations.description)

    data_1, times_1 = raw_1[:, :]
    data_2, times_2 = raw_2[:, :]
    assert_array_equal(data_1, data_2)
    assert_array_equal(times_1, times_2)

    raw_bids = read_raw_fif(split_fname_bids_part1)
    data_bids, times_bids = raw_bids[:, :]
    assert_array_equal(data_1, data_bids)
    assert_array_equal(times_1, times_bids)

    # test the case where we only end up with one buffer to write
    # (GH#3210). These tests rely on writing meas info and annotations
    # taking up a certain number of bytes, so if we change those functions
    # somehow, the numbers below for e.g. split_size might need to be
    # adjusted.
    raw_crop = raw_1.copy().crop(0, 5)
    raw_crop.set_annotations(Annotations([2.], [5.5], 'test'),
                             emit_warning=False)
    with pytest.raises(ValueError,
                       match='after writing measurement information'):
        raw_crop.save(split_fname, split_size='1MB',  # too small a size
                      buffer_size_sec=1., overwrite=True)
    with pytest.raises(ValueError,
                       match='too large for the given split size'):
        raw_crop.save(split_fname,
                      split_size=3003000,  # still too small, now after Info
                      buffer_size_sec=1., overwrite=True)
    # just barely big enough here; the right size to write exactly one buffer
    # at a time so we hit GH#3210 if we aren't careful
    raw_crop.save(split_fname, split_size='4.5MB',
                  buffer_size_sec=1., overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_allclose(raw_crop[:][0], raw_read[:][0], atol=1e-20)

    # Check our buffer arithmetic

    # 1 buffer required
    raw_crop = raw_1.copy().crop(0, 1)
    raw_crop.save(split_fname, buffer_size_sec=1., overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_array_equal(np.diff(raw_read._raw_extras[0]['bounds']), (301,))
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    # 2 buffers required
    raw_crop.save(split_fname, buffer_size_sec=0.5, overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_array_equal(np.diff(raw_read._raw_extras[0]['bounds']), (151, 150))
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    # 2 buffers required
    raw_crop.save(split_fname,
                  buffer_size_sec=1. - 1.01 / raw_crop.info['sfreq'],
                  overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_array_equal(np.diff(raw_read._raw_extras[0]['bounds']), (300, 1))
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    raw_crop.save(split_fname,
                  buffer_size_sec=1. - 2.01 / raw_crop.info['sfreq'],
                  overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_array_equal(np.diff(raw_read._raw_extras[0]['bounds']), (299, 2))
    assert_allclose(raw_crop[:][0], raw_read[:][0])


def test_load_bad_channels(tmpdir):
    """Test reading/writing of bad channels."""
    # Load correctly marked file (manually done in mne_process_raw)
    raw_marked = read_raw_fif(fif_bad_marked_fname)
    correct_bads = raw_marked.info['bads']
    raw = read_raw_fif(test_fif_fname)
    # Make sure it starts clean
    assert_array_equal(raw.info['bads'], [])

    # Test normal case
    raw.load_bad_channels(bad_file_works)
    # Write it out, read it in, and check
    raw.save(tmpdir.join('foo_raw.fif'))
    raw_new = read_raw_fif(tmpdir.join('foo_raw.fif'))
    assert correct_bads == raw_new.info['bads']
    # Reset it
    raw.info['bads'] = []

    # Test bad case
    pytest.raises(ValueError, raw.load_bad_channels, bad_file_wrong)

    # Test forcing the bad case
    with pytest.warns(RuntimeWarning, match='1 bad channel'):
        raw.load_bad_channels(bad_file_wrong, force=True)
        # write it out, read it in, and check
    raw.save(tmpdir.join('foo_raw.fif'), overwrite=True)
    raw_new = read_raw_fif(tmpdir.join('foo_raw.fif'))
    assert correct_bads == raw_new.info['bads']

    # Check that bad channels are cleared
    raw.load_bad_channels(None)
    raw.save(tmpdir.join('foo_raw.fif'), overwrite=True)
    raw_new = read_raw_fif(tmpdir.join('foo_raw.fif'))
    assert raw_new.info['bads'] == []


@pytest.mark.slowtest
@testing.requires_testing_data
def test_io_raw(tmpdir):
    """Test IO for raw data (Neuromag)."""
    rng = np.random.RandomState(0)
    # test unicode io
    for chars in [u'äöé', 'a']:
        with read_raw_fif(fif_fname) as r:
            assert ('Raw' in repr(r))
            assert (op.basename(fif_fname) in repr(r))
            r.info['description'] = chars
            temp_file = tmpdir.join('raw.fif')
            r.save(temp_file, overwrite=True)
            with read_raw_fif(temp_file) as r2:
                desc2 = r2.info['description']
            assert desc2 == chars

    # Let's construct a simple test for IO first
    raw = read_raw_fif(fif_fname).crop(0, 3.5)
    raw.load_data()
    # put in some data that we know the values of
    data = rng.randn(raw._data.shape[0], raw._data.shape[1])
    raw._data[:, :] = data
    # save it somewhere
    fname = tmpdir.join('test_copy_raw.fif')
    raw.save(fname, buffer_size_sec=1.0)
    # read it in, make sure the whole thing matches
    raw = read_raw_fif(fname)
    assert_allclose(data, raw[:, :][0], rtol=1e-6, atol=1e-20)
    # let's read portions across the 1-sec tag boundary, too
    inds = raw.time_as_index([1.75, 2.25])
    sl = slice(inds[0], inds[1])
    assert_allclose(data[:, sl], raw[:, sl][0], rtol=1e-6, atol=1e-20)


@pytest.mark.parametrize('fname_in, fname_out', [
    (test_fif_fname, 'raw.fif'),
    (test_fif_gz_fname, 'raw.fif.gz'),
    (ctf_fname, 'raw.fif')])
def test_io_raw_additional(fname_in, fname_out, tmpdir):
    """Test IO for raw data (Neuromag + CTF + gz)."""
    fname_out = tmpdir.join(fname_out)
    raw = read_raw_fif(fname_in)

    nchan = raw.info['nchan']
    ch_names = raw.info['ch_names']
    meg_channels_idx = [k for k in range(nchan)
                        if ch_names[k][0] == 'M']
    n_channels = 100
    meg_channels_idx = meg_channels_idx[:n_channels]
    start, stop = raw.time_as_index([0, 5], use_rounding=True)
    data, times = raw[meg_channels_idx, start:(stop + 1)]
    meg_ch_names = [ch_names[k] for k in meg_channels_idx]

    # Set up pick list: MEG + STI 014 - bad channels
    include = ['STI 014']
    include += meg_ch_names
    picks = pick_types(raw.info, meg=True, eeg=False, stim=True,
                       misc=True, ref_meg=True, include=include,
                       exclude='bads')

    # Writing with drop_small_buffer True
    raw.save(fname_out, picks, tmin=0, tmax=4, buffer_size_sec=3,
             drop_small_buffer=True, overwrite=True)
    raw2 = read_raw_fif(fname_out)

    sel = pick_channels(raw2.ch_names, meg_ch_names)
    data2, times2 = raw2[sel, :]
    assert (times2.max() <= 3)

    # Writing
    raw.save(fname_out, picks, tmin=0, tmax=5, overwrite=True)

    if fname_in in (fif_fname, fif_fname + '.gz'):
        assert len(raw.info['dig']) == 146

    raw2 = read_raw_fif(fname_out)

    sel = pick_channels(raw2.ch_names, meg_ch_names)
    data2, times2 = raw2[sel, :]

    assert_allclose(data, data2, rtol=1e-6, atol=1e-20)
    assert_allclose(times, times2)
    assert_allclose(raw.info['sfreq'], raw2.info['sfreq'], rtol=1e-5)

    # check transformations
    for trans in ['dev_head_t', 'dev_ctf_t', 'ctf_head_t']:
        if raw.info[trans] is None:
            assert (raw2.info[trans] is None)
        else:
            assert_array_equal(raw.info[trans]['trans'],
                               raw2.info[trans]['trans'])

            # check transformation 'from' and 'to'
            if trans.startswith('dev'):
                from_id = FIFF.FIFFV_COORD_DEVICE
            else:
                from_id = FIFF.FIFFV_MNE_COORD_CTF_HEAD
            if trans[4:8] == 'head':
                to_id = FIFF.FIFFV_COORD_HEAD
            else:
                to_id = FIFF.FIFFV_MNE_COORD_CTF_HEAD
            for raw_ in [raw, raw2]:
                assert raw_.info[trans]['from'] == from_id
                assert raw_.info[trans]['to'] == to_id

    if fname_in == fif_fname or fname_in == fif_fname + '.gz':
        assert_allclose(raw.info['dig'][0]['r'], raw2.info['dig'][0]['r'])

    # test warnings on bad filenames
    raw_badname = tmpdir.join('test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='raw.fif'):
        raw.save(raw_badname)
    with pytest.warns(RuntimeWarning, match='raw.fif'):
        read_raw_fif(raw_badname)


@testing.requires_testing_data
def test_io_complex(tmpdir):
    """Test IO with complex data types."""
    rng = np.random.RandomState(0)
    dtypes = [np.complex64, np.complex128]

    raw = _test_raw_reader(partial(read_raw_fif),
                           fname=fif_fname)
    picks = np.arange(5)
    start, stop = raw.time_as_index([0, 5])

    data_orig, _ = raw[picks, start:stop]

    for di, dtype in enumerate(dtypes):
        imag_rand = np.array(1j * rng.randn(data_orig.shape[0],
                                            data_orig.shape[1]), dtype)

        raw_cp = raw.copy()
        raw_cp._data = np.array(raw_cp._data, dtype)
        raw_cp._data[picks, start:stop] += imag_rand
        with pytest.warns(RuntimeWarning, match='Saving .* complex data.'):
            raw_cp.save(tmpdir.join('raw.fif'), picks, tmin=0, tmax=5,
                        overwrite=True)

        raw2 = read_raw_fif(tmpdir.join('raw.fif'))
        raw2_data, _ = raw2[picks, :]
        n_samp = raw2_data.shape[1]
        assert_allclose(raw2_data[:, :n_samp], raw_cp._data[picks, :n_samp])
        # with preloading
        raw2 = read_raw_fif(tmpdir.join('raw.fif'), preload=True)
        raw2_data, _ = raw2[picks, :]
        n_samp = raw2_data.shape[1]
        assert_allclose(raw2_data[:, :n_samp], raw_cp._data[picks, :n_samp])


@testing.requires_testing_data
def test_getitem():
    """Test getitem/indexing of Raw."""
    for preload in [False, True, 'memmap.dat']:
        raw = read_raw_fif(fif_fname, preload=preload)
        data, times = raw[0, :]
        data1, times1 = raw[0]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)
        data, times = raw[0:2, :]
        data1, times1 = raw[0:2]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)
        data1, times1 = raw[[0, 1]]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)
        assert_array_equal(raw[raw.ch_names[0]][0][0], raw[0][0][0])
        assert_array_equal(
            raw[-10:-1, :][0],
            raw[len(raw.ch_names) - 10:len(raw.ch_names) - 1, :][0])
        with pytest.raises(ValueError, match='No appropriate channels'):
            raw[slice(-len(raw.ch_names) - 1), slice(None)]
        with pytest.raises(ValueError, match='must be'):
            raw[-1000]


@testing.requires_testing_data
def test_proj(tmpdir):
    """Test SSP proj operations."""
    for proj in [True, False]:
        raw = read_raw_fif(fif_fname, preload=False)
        if proj:
            raw.apply_proj()
        assert (all(p['active'] == proj for p in raw.info['projs']))

        data, times = raw[0:2, :]
        data1, times1 = raw[0:2]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)

        # test adding / deleting proj
        if proj:
            pytest.raises(ValueError, raw.add_proj, [],
                          {'remove_existing': True})
            pytest.raises(ValueError, raw.del_proj, 0)
        else:
            projs = deepcopy(raw.info['projs'])
            n_proj = len(raw.info['projs'])
            raw.del_proj(0)
            assert len(raw.info['projs']) == n_proj - 1
            raw.add_proj(projs, remove_existing=False)
            # Test that already existing projections are not added.
            assert len(raw.info['projs']) == n_proj
            raw.add_proj(projs[:-1], remove_existing=True)
            assert len(raw.info['projs']) == n_proj - 1

    # test apply_proj() with and without preload
    for preload in [True, False]:
        raw = read_raw_fif(fif_fname, preload=preload)
        data, times = raw[:, 0:2]
        raw.apply_proj()
        data_proj_1 = np.dot(raw._projector, data)

        # load the file again without proj
        raw = read_raw_fif(fif_fname, preload=preload)

        # write the file with proj. activated, make sure proj has been applied
        raw.save(tmpdir.join('raw.fif'), proj=True, overwrite=True)
        raw2 = read_raw_fif(tmpdir.join('raw.fif'))
        data_proj_2, _ = raw2[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert (all(p['active'] for p in raw2.info['projs']))

        # read orig file with proj. active
        raw2 = read_raw_fif(fif_fname, preload=preload)
        raw2.apply_proj()
        data_proj_2, _ = raw2[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert (all(p['active'] for p in raw2.info['projs']))

        # test that apply_proj works
        raw.apply_proj()
        data_proj_2, _ = raw[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert_allclose(data_proj_2, np.dot(raw._projector, data_proj_2))

    out_fname = tmpdir.join('test_raw.fif')
    raw = read_raw_fif(test_fif_fname, preload=True).crop(0, 0.002)
    raw.pick_types(meg=False, eeg=True)
    raw.info['projs'] = [raw.info['projs'][-1]]
    raw._data.fill(0)
    raw._data[-1] = 1.
    raw.save(out_fname)
    raw = read_raw_fif(out_fname, preload=False)
    raw.apply_proj()
    assert_allclose(raw[:, :][0][:1], raw[0, :][0])


@testing.requires_testing_data
@pytest.mark.parametrize('preload', [False, True, 'memmap.dat'])
def test_preload_modify(preload, tmpdir):
    """Test preloading and modifying data."""
    rng = np.random.RandomState(0)
    raw = read_raw_fif(fif_fname, preload=preload)

    nsamp = raw.last_samp - raw.first_samp + 1
    picks = pick_types(raw.info, meg='grad', exclude='bads')

    data = rng.randn(len(picks), nsamp // 2)

    try:
        raw[picks, :nsamp // 2] = data
    except RuntimeError:
        if not preload:
            return
        else:
            raise

    tmp_fname = tmpdir.join('raw.fif')
    raw.save(tmp_fname, overwrite=True)

    raw_new = read_raw_fif(tmp_fname)
    data_new, _ = raw_new[picks, :nsamp // 2]

    assert_allclose(data, data_new)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_filter():
    """Test filtering (FIR and IIR) and Raw.apply_function interface."""
    raw = read_raw_fif(fif_fname).crop(0, 7)
    raw.load_data()
    sig_dec_notch = 12
    sig_dec_notch_fit = 12
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    trans = 2.0
    filter_params = dict(picks=picks, filter_length='auto',
                         h_trans_bandwidth=trans, l_trans_bandwidth=trans,
                         fir_design='firwin')
    raw_lp = raw.copy().filter(None, 8.0, **filter_params)
    raw_hp = raw.copy().filter(16.0, None, **filter_params)
    raw_bp = raw.copy().filter(8.0 + trans, 16.0 - trans, **filter_params)
    raw_bs = raw.copy().filter(16.0, 8.0, **filter_params)

    data, _ = raw[picks, :]

    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]

    tols = dict(atol=1e-20, rtol=1e-5)
    assert_allclose(bs_data, lp_data + hp_data, **tols)
    assert_allclose(data, lp_data + bp_data + hp_data, **tols)
    assert_allclose(data, bp_data + bs_data, **tols)

    filter_params_iir = dict(picks=picks, n_jobs=2, method='iir',
                             iir_params=dict(output='ba'))
    raw_lp_iir = raw.copy().filter(None, 4.0, **filter_params_iir)
    raw_hp_iir = raw.copy().filter(8.0, None, **filter_params_iir)
    raw_bp_iir = raw.copy().filter(4.0, 8.0, **filter_params_iir)
    del filter_params_iir
    lp_data_iir, _ = raw_lp_iir[picks, :]
    hp_data_iir, _ = raw_hp_iir[picks, :]
    bp_data_iir, _ = raw_bp_iir[picks, :]
    summation = lp_data_iir + hp_data_iir + bp_data_iir
    assert_array_almost_equal(data[:, 100:-100], summation[:, 100:-100], 11)

    # make sure we didn't touch other channels
    data, _ = raw[picks_meg[4:], :]
    bp_data, _ = raw_bp[picks_meg[4:], :]
    assert_array_equal(data, bp_data)
    bp_data_iir, _ = raw_bp_iir[picks_meg[4:], :]
    assert_array_equal(data, bp_data_iir)

    # ... and that inplace changes are inplace
    raw_copy = raw.copy()
    assert np.may_share_memory(raw._data, raw._data)
    assert not np.may_share_memory(raw_copy._data, raw._data)
    # this could be assert_array_equal but we do this to mirror the call below
    assert (raw._data[0] == raw_copy._data[0]).all()
    raw_copy.filter(None, 20., n_jobs=2, **filter_params)
    assert not (raw._data[0] == raw_copy._data[0]).all()
    assert_array_equal(raw.copy().filter(None, 20., **filter_params)._data,
                       raw_copy._data)

    # do a very simple check on line filtering
    raw_bs = raw.copy().filter(60.0 + trans, 60.0 - trans, **filter_params)
    data_bs, _ = raw_bs[picks, :]
    raw_notch = raw.copy().notch_filter(
        60.0, picks=picks, n_jobs=2, method='fir',
        trans_bandwidth=2 * trans)
    data_notch, _ = raw_notch[picks, :]
    assert_array_almost_equal(data_bs, data_notch, sig_dec_notch)

    # now use the sinusoidal fitting
    raw_notch = raw.copy().notch_filter(
        None, picks=picks, n_jobs=2, method='spectrum_fit')
    data_notch, _ = raw_notch[picks, :]
    data, _ = raw[picks, :]
    assert_array_almost_equal(data, data_notch, sig_dec_notch_fit)

    # filter should set the "lowpass" and "highpass" parameters
    raw = RawArray(np.random.randn(3, 1000),
                   create_info(3, 1000., ['eeg'] * 2 + ['stim']))
    raw.info['lowpass'] = raw.info['highpass'] = None
    for kind in ('none', 'lowpass', 'highpass', 'bandpass', 'bandstop'):
        print(kind)
        h_freq = l_freq = None
        if kind in ('lowpass', 'bandpass'):
            h_freq = 70
        if kind in ('highpass', 'bandpass'):
            l_freq = 30
        if kind == 'bandstop':
            l_freq, h_freq = 70, 30
        assert (raw.info['lowpass'] is None)
        assert (raw.info['highpass'] is None)
        kwargs = dict(l_trans_bandwidth=20, h_trans_bandwidth=20,
                      filter_length='auto', phase='zero', fir_design='firwin')
        raw_filt = raw.copy().filter(l_freq, h_freq, picks=np.arange(1),
                                     **kwargs)
        assert (raw.info['lowpass'] is None)
        assert (raw.info['highpass'] is None)
        raw_filt = raw.copy().filter(l_freq, h_freq, **kwargs)
        wanted_h = h_freq if kind != 'bandstop' else None
        wanted_l = l_freq if kind != 'bandstop' else None
        assert raw_filt.info['lowpass'] == wanted_h
        assert raw_filt.info['highpass'] == wanted_l
        # Using all data channels should still set the params (GH#3259)
        raw_filt = raw.copy().filter(l_freq, h_freq, picks=np.arange(2),
                                     **kwargs)
        assert raw_filt.info['lowpass'] == wanted_h
        assert raw_filt.info['highpass'] == wanted_l


def test_filter_picks():
    """Test filtering default channel picks."""
    ch_types = ['mag', 'grad', 'eeg', 'seeg', 'misc', 'stim', 'ecog', 'hbo',
                'hbr']
    info = create_info(ch_names=ch_types, ch_types=ch_types, sfreq=256)
    raw = RawArray(data=np.zeros((len(ch_types), 1000)), info=info)

    # -- Deal with meg mag grad and fnirs exceptions
    ch_types = ('misc', 'stim', 'meg', 'eeg', 'seeg', 'ecog')

    # -- Filter data channels
    for ch_type in ('mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', 'hbr'):
        picks = {ch: ch == ch_type for ch in ch_types}
        picks['meg'] = ch_type if ch_type in ('mag', 'grad') else False
        picks['fnirs'] = ch_type if ch_type in ('hbo', 'hbr') else False
        raw_ = raw.copy().pick_types(**picks)
        raw_.filter(10, 30, fir_design='firwin')

    # -- Error if no data channel
    for ch_type in ('misc', 'stim'):
        picks = {ch: ch == ch_type for ch in ch_types}
        raw_ = raw.copy().pick_types(**picks)
        pytest.raises(ValueError, raw_.filter, 10, 30)


@testing.requires_testing_data
def test_crop():
    """Test cropping raw files."""
    # split a concatenated file to test a difficult case
    raw = concatenate_raws([read_raw_fif(f)
                            for f in [fif_fname, fif_fname]])
    split_size = 10.  # in seconds
    sfreq = raw.info['sfreq']
    nsamp = (raw.last_samp - raw.first_samp + 1)

    # do an annoying case (off-by-one splitting)
    tmins = np.r_[1., np.round(np.arange(0., nsamp - 1, split_size * sfreq))]
    tmins = np.sort(tmins)
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp - 1]))
    tmaxs /= sfreq
    tmins /= sfreq
    raws = [None] * len(tmins)
    for ri, (tmin, tmax) in enumerate(zip(tmins, tmaxs)):
        raws[ri] = raw.copy().crop(tmin, tmax)
        if ri < len(tmins) - 1:
            assert_allclose(
                raws[ri].times,
                raw.copy().crop(tmin, tmins[ri + 1], include_tmax=False).times)
        assert raws[ri]
    all_raw_2 = concatenate_raws(raws, preload=False)
    assert raw.first_samp == all_raw_2.first_samp
    assert raw.last_samp == all_raw_2.last_samp
    assert_array_equal(raw[:, :][0], all_raw_2[:, :][0])

    tmins = np.round(np.arange(0., nsamp - 1, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp - 1]))
    tmaxs /= sfreq
    tmins /= sfreq

    # going in revere order so the last fname is the first file (need it later)
    raws = [None] * len(tmins)
    for ri, (tmin, tmax) in enumerate(zip(tmins, tmaxs)):
        raws[ri] = raw.copy().crop(tmin, tmax)
    # test concatenation of split file
    all_raw_1 = concatenate_raws(raws, preload=False)

    all_raw_2 = raw.copy().crop(0, None)
    for ar in [all_raw_1, all_raw_2]:
        assert raw.first_samp == ar.first_samp
        assert raw.last_samp == ar.last_samp
        assert_array_equal(raw[:, :][0], ar[:, :][0])

    # test shape consistency of cropped raw
    data = np.zeros((1, 1002001))
    info = create_info(1, 1000)
    raw = RawArray(data, info)
    for tmin in range(0, 1001, 100):
        raw1 = raw.copy().crop(tmin=tmin, tmax=tmin + 2)
        assert raw1[:][0].shape == (1, 2001)

    # degenerate
    with pytest.raises(ValueError, match='No samples.*when include_tmax=Fals'):
        raw.crop(0, 0, include_tmax=False)


@testing.requires_testing_data
def test_resample(tmpdir):
    """Test resample (with I/O and multiple files)."""
    raw = read_raw_fif(fif_fname).crop(0, 3)
    raw.load_data()
    raw_resamp = raw.copy()
    sfreq = raw.info['sfreq']
    # test parallel on upsample
    raw_resamp.resample(sfreq * 2, n_jobs=2, npad='auto')
    assert raw_resamp.n_times == len(raw_resamp.times)
    raw_resamp.save(tmpdir.join('raw_resamp-raw.fif'))
    raw_resamp = read_raw_fif(tmpdir.join('raw_resamp-raw.fif'),
                              preload=True)
    assert sfreq == raw_resamp.info['sfreq'] / 2
    assert raw.n_times == raw_resamp.n_times // 2
    assert raw_resamp._data.shape[1] == raw_resamp.n_times
    assert raw._data.shape[0] == raw_resamp._data.shape[0]
    # test non-parallel on downsample
    raw_resamp.resample(sfreq, n_jobs=1, npad='auto')
    assert raw_resamp.info['sfreq'] == sfreq
    assert raw._data.shape == raw_resamp._data.shape
    assert raw.first_samp == raw_resamp.first_samp
    assert raw.last_samp == raw.last_samp
    # upsampling then downsampling doubles resampling error, but this still
    # works (hooray). Note that the stim channels had to be sub-sampled
    # without filtering to be accurately preserved
    # note we have to treat MEG and EEG+STIM channels differently (tols)
    assert_allclose(raw._data[:306, 200:-200],
                    raw_resamp._data[:306, 200:-200],
                    rtol=1e-2, atol=1e-12)
    assert_allclose(raw._data[306:, 200:-200],
                    raw_resamp._data[306:, 200:-200],
                    rtol=1e-2, atol=1e-7)

    # now check multiple file support w/resampling, as order of operations
    # (concat, resample) should not affect our data
    raw1 = raw.copy()
    raw2 = raw.copy()
    raw3 = raw.copy()
    raw4 = raw.copy()
    raw1 = concatenate_raws([raw1, raw2])
    raw1.resample(10., npad='auto')
    raw3.resample(10., npad='auto')
    raw4.resample(10., npad='auto')
    raw3 = concatenate_raws([raw3, raw4])
    assert_array_equal(raw1._data, raw3._data)
    assert_array_equal(raw1._first_samps, raw3._first_samps)
    assert_array_equal(raw1._last_samps, raw3._last_samps)
    assert_array_equal(raw1._raw_lengths, raw3._raw_lengths)
    assert raw1.first_samp == raw3.first_samp
    assert raw1.last_samp == raw3.last_samp
    assert raw1.info['sfreq'] == raw3.info['sfreq']

    # test resampling of stim channel

    # basic decimation
    stim = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    assert_allclose(raw.resample(8., npad='auto')._data,
                    [[1, 1, 0, 0, 1, 1, 0, 0]])

    # decimation of multiple stim channels
    raw = RawArray(2 * [stim], create_info(2, len(stim), 2 * ['stim']))
    assert_allclose(raw.resample(8., npad='auto', verbose='error')._data,
                    [[1, 1, 0, 0, 1, 1, 0, 0],
                     [1, 1, 0, 0, 1, 1, 0, 0]])

    # decimation that could potentially drop events if the decimation is
    # done naively
    stim = [0, 0, 0, 1, 1, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    assert_allclose(raw.resample(4., npad='auto')._data,
                    [[0, 1, 1, 0]])

    # two events are merged in this case (warning)
    stim = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    with pytest.warns(RuntimeWarning, match='become unreliable'):
        raw.resample(8., npad='auto')

    # events are dropped in this case (warning)
    stim = [0, 1, 1, 0, 0, 1, 1, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    with pytest.warns(RuntimeWarning, match='become unreliable'):
        raw.resample(4., npad='auto')

    # test resampling events: this should no longer give a warning
    # we often have first_samp != 0, include it here too
    stim = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  # an event at end
    # test is on half the sfreq, but should work with trickier ones too
    o_sfreq, sfreq_ratio = len(stim), 0.5
    n_sfreq = o_sfreq * sfreq_ratio
    first_samp = len(stim) // 2
    raw = RawArray([stim], create_info(1, o_sfreq, ['stim']),
                   first_samp=first_samp)
    events = find_events(raw)
    raw, events = raw.resample(n_sfreq, events=events, npad='auto')
    # Try index into raw.times with resampled events:
    raw.times[events[:, 0] - raw.first_samp]
    n_fsamp = int(first_samp * sfreq_ratio)  # how it's calc'd in base.py
    # NB np.round used for rounding event times, which has 0.5 as corner case:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.around.html
    assert_array_equal(
        events,
        np.array([[np.round(1 * sfreq_ratio) + n_fsamp, 0, 1],
                  [np.round(10 * sfreq_ratio) + n_fsamp, 0, 1],
                  [np.minimum(np.round(15 * sfreq_ratio),
                              raw._data.shape[1] - 1) + n_fsamp, 0, 1]]))

    # test copy flag
    stim = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    raw_resampled = raw.copy().resample(4., npad='auto')
    assert (raw_resampled is not raw)
    raw_resampled = raw.resample(4., npad='auto')
    assert (raw_resampled is raw)

    # resample should still work even when no stim channel is present
    raw = RawArray(np.random.randn(1, 100), create_info(1, 100, ['eeg']))
    raw.info['lowpass'] = 50.
    raw.resample(10, npad='auto')
    assert raw.info['lowpass'] == 5.
    assert len(raw) == 10


@testing.requires_testing_data
def test_hilbert():
    """Test computation of analytic signal using hilbert."""
    raw = read_raw_fif(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    raw_filt = raw.copy()
    raw_filt.filter(10, 20, picks=picks, l_trans_bandwidth='auto',
                    h_trans_bandwidth='auto', filter_length='auto',
                    phase='zero', fir_window='blackman', fir_design='firwin')
    raw_filt_2 = raw_filt.copy()

    raw2 = raw.copy()
    raw3 = raw.copy()
    raw.apply_hilbert(picks, n_fft='auto')
    raw2.apply_hilbert(picks, n_fft='auto', envelope=True)

    # Test custom n_fft
    raw_filt.apply_hilbert(picks, n_fft='auto')
    n_fft = 2 ** int(np.ceil(np.log2(raw_filt_2.n_times + 1000)))
    raw_filt_2.apply_hilbert(picks, n_fft=n_fft)
    assert raw_filt._data.shape == raw_filt_2._data.shape
    assert_allclose(raw_filt._data[:, 50:-50], raw_filt_2._data[:, 50:-50],
                    atol=1e-13, rtol=1e-2)
    with pytest.raises(ValueError, match='n_fft.*must be at least the number'):
        raw3.apply_hilbert(picks, n_fft=raw3.n_times - 100)

    env = np.abs(raw._data[picks, :])
    assert_allclose(env, raw2._data[picks, :], rtol=1e-2, atol=1e-13)


@testing.requires_testing_data
def test_raw_copy():
    """Test Raw copy."""
    raw = read_raw_fif(fif_fname, preload=True)
    data, _ = raw[:, :]
    copied = raw.copy()
    copied_data, _ = copied[:, :]
    assert_array_equal(data, copied_data)
    assert sorted(raw.__dict__.keys()) == sorted(copied.__dict__.keys())

    raw = read_raw_fif(fif_fname, preload=False)
    data, _ = raw[:, :]
    copied = raw.copy()
    copied_data, _ = copied[:, :]
    assert_array_equal(data, copied_data)
    assert sorted(raw.__dict__.keys()) == sorted(copied.__dict__.keys())


@requires_pandas
def test_to_data_frame():
    """Test raw Pandas exporter."""
    from pandas import Timedelta
    raw = read_raw_fif(test_fif_fname, preload=True)
    _, times = raw[0, :10]
    df = raw.to_data_frame(index='time')
    assert ((df.columns == raw.ch_names).all())
    assert_array_equal(np.round(times * 1e3), df.index.values[:10])
    df = raw.to_data_frame(index=None)
    assert ('time' in df.columns)
    assert_array_equal(df.values[:, 1], raw._data[0] * 1e13)
    assert_array_equal(df.values[:, 3], raw._data[2] * 1e15)
    # test long format
    df_long = raw.to_data_frame(long_format=True)
    assert(len(df_long) == raw.get_data().size)
    expected = ('time', 'channel', 'ch_type', 'value')
    assert set(expected) == set(df_long.columns)
    # test bad time format
    with pytest.raises(ValueError, match='not a valid time format. Valid'):
        raw.to_data_frame(time_format='foo')
    # test time format error handling
    raw.set_meas_date(None)
    with pytest.warns(RuntimeWarning, match='Cannot convert to Datetime when'):
        df = raw.to_data_frame(time_format='datetime')
    assert isinstance(df['time'].iloc[0], Timedelta)


@requires_pandas
@pytest.mark.parametrize('time_format', (None, 'ms', 'timedelta', 'datetime'))
def test_to_data_frame_time_format(time_format):
    """Test time conversion in epochs Pandas exporter."""
    from pandas import Timedelta, Timestamp
    raw = read_raw_fif(test_fif_fname, preload=True)
    # test time_format
    df = raw.to_data_frame(time_format=time_format)
    dtypes = {None: np.float64, 'ms': np.int64, 'timedelta': Timedelta,
              'datetime': Timestamp}
    assert isinstance(df['time'].iloc[0], dtypes[time_format])


def test_add_channels():
    """Test raw splitting / re-appending channel types."""
    rng = np.random.RandomState(0)
    raw = read_raw_fif(test_fif_fname).crop(0, 1).load_data()
    raw_nopre = read_raw_fif(test_fif_fname, preload=False)
    raw_eeg_meg = raw.copy().pick_types(meg=True, eeg=True)
    raw_eeg = raw.copy().pick_types(meg=False, eeg=True)
    raw_meg = raw.copy().pick_types(meg=True, eeg=False)
    raw_stim = raw.copy().pick_types(meg=False, eeg=False, stim=True)
    raw_new = raw_meg.copy().add_channels([raw_eeg, raw_stim])
    assert (
        all(ch in raw_new.ch_names
            for ch in list(raw_stim.ch_names) + list(raw_meg.ch_names))
    )
    raw_new = raw_meg.copy().add_channels([raw_eeg])

    assert (ch in raw_new.ch_names for ch in raw.ch_names)
    assert_array_equal(raw_new[:, :][0], raw_eeg_meg[:, :][0])
    assert_array_equal(raw_new[:, :][1], raw[:, :][1])
    assert (all(ch not in raw_new.ch_names for ch in raw_stim.ch_names))

    # Testing force updates
    raw_arr_info = create_info(['1', '2'], raw_meg.info['sfreq'], 'eeg')
    orig_head_t = raw_arr_info['dev_head_t']
    raw_arr = rng.randn(2, raw_eeg.n_times)
    raw_arr = RawArray(raw_arr, raw_arr_info)
    # This should error because of conflicts in Info
    raw_arr.info['dev_head_t'] = orig_head_t
    with pytest.raises(ValueError, match='mutually inconsistent dev_head_t'):
        raw_meg.copy().add_channels([raw_arr])
    raw_meg.copy().add_channels([raw_arr], force_update_info=True)
    # Make sure that values didn't get overwritten
    assert_object_equal(raw_arr.info['dev_head_t'], orig_head_t)

    # Now test errors
    raw_badsf = raw_eeg.copy()
    raw_badsf.info['sfreq'] = 3.1415927
    raw_eeg.crop(.5)

    pytest.raises(RuntimeError, raw_meg.add_channels, [raw_nopre])
    pytest.raises(RuntimeError, raw_meg.add_channels, [raw_badsf])
    pytest.raises(AssertionError, raw_meg.add_channels, [raw_eeg])
    pytest.raises(ValueError, raw_meg.add_channels, [raw_meg])
    pytest.raises(TypeError, raw_meg.add_channels, raw_badsf)


@testing.requires_testing_data
def test_save(tmpdir):
    """Test saving raw."""
    raw = read_raw_fif(fif_fname, preload=False)
    # can't write over file being read
    pytest.raises(ValueError, raw.save, fif_fname)
    raw = read_raw_fif(fif_fname, preload=True)
    # can't overwrite file without overwrite=True
    pytest.raises(IOError, raw.save, fif_fname)

    # test abspath support and annotations
    orig_time = _dt_to_stamp(raw.info['meas_date'])[0] + raw._first_time
    annot = Annotations([10], [5], ['test'], orig_time=orig_time)
    raw.set_annotations(annot)
    annot = raw.annotations
    new_fname = tmpdir.join('break_raw.fif')
    raw.save(new_fname, overwrite=True)
    new_raw = read_raw_fif(new_fname, preload=False)
    pytest.raises(ValueError, new_raw.save, new_fname)
    assert_array_almost_equal(annot.onset, new_raw.annotations.onset)
    assert_array_equal(annot.duration, new_raw.annotations.duration)
    assert_array_equal(annot.description, new_raw.annotations.description)
    assert annot.orig_time == new_raw.annotations.orig_time


@testing.requires_testing_data
def test_annotation_crop(tmpdir):
    """Test annotation sync after cropping and concatenating."""
    annot = Annotations([5., 11., 15.], [2., 1., 3.], ['test', 'test', 'test'])
    raw = read_raw_fif(fif_fname, preload=False)
    raw.set_annotations(annot)
    r1 = raw.copy().crop(2.5, 7.5)
    r2 = raw.copy().crop(12.5, 17.5)
    r3 = raw.copy().crop(10., 12.)
    raw = concatenate_raws([r1, r2, r3])  # segments reordered
    assert_and_remove_boundary_annot(raw, 2)
    onsets = raw.annotations.onset
    durations = raw.annotations.duration
    # 2*5s clips combined with annotations at 2.5s + 2s clip, annotation at 1s
    assert_array_almost_equal(onsets[:3], [47.95, 52.95, 56.46], decimal=2)
    assert_array_almost_equal([2., 2.5, 1.], durations[:3], decimal=2)

    # test annotation clipping
    orig_time = _dt_to_stamp(raw.info['meas_date'])
    orig_time = orig_time[0] + orig_time[1] * 1e-6 + raw._first_time - 1.
    annot = Annotations([0., raw.times[-1]], [2., 2.], 'test', orig_time)
    with pytest.warns(RuntimeWarning, match='Limited .* expanding outside'):
        raw.set_annotations(annot)
    assert_allclose(raw.annotations.duration,
                    [1., 1. + 1. / raw.info['sfreq']], atol=1e-3)

    # make sure we can overwrite the file we loaded when preload=True
    new_fname = tmpdir.join('break_raw.fif')
    raw.save(new_fname)
    new_raw = read_raw_fif(new_fname, preload=True)
    new_raw.save(new_fname, overwrite=True)


@testing.requires_testing_data
def test_with_statement():
    """Test with statement."""
    for preload in [True, False]:
        with read_raw_fif(fif_fname, preload=preload) as raw_:
            print(raw_)


def test_compensation_raw(tmpdir):
    """Test Raw compensation."""
    raw_3 = read_raw_fif(ctf_comp_fname)
    assert raw_3.compensation_grade == 3
    data_3, times = raw_3[:, :]

    # data come with grade 3
    for ii in range(2):
        raw_3_new = raw_3.copy()
        if ii == 0:
            raw_3_new.load_data()
        raw_3_new.apply_gradient_compensation(3)
        assert raw_3_new.compensation_grade == 3
        data_new, times_new = raw_3_new[:, :]
        assert_array_equal(times, times_new)
        assert_array_equal(data_3, data_new)

    # change to grade 0
    raw_0 = raw_3.copy().apply_gradient_compensation(0)
    assert raw_0.compensation_grade == 0
    data_0, times_new = raw_0[:, :]
    assert_array_equal(times, times_new)
    assert (np.mean(np.abs(data_0 - data_3)) > 1e-12)
    # change to grade 1
    raw_1 = raw_0.copy().apply_gradient_compensation(1)
    assert raw_1.compensation_grade == 1
    data_1, times_new = raw_1[:, :]
    assert_array_equal(times, times_new)
    assert (np.mean(np.abs(data_1 - data_3)) > 1e-12)
    pytest.raises(ValueError, raw_1.apply_gradient_compensation, 33)
    raw_bad = raw_0.copy()
    raw_bad.add_proj(compute_proj_raw(raw_0, duration=0.5, verbose='error'))
    raw_bad.apply_proj()
    pytest.raises(RuntimeError, raw_bad.apply_gradient_compensation, 1)
    # with preload
    tols = dict(rtol=1e-12, atol=1e-25)
    raw_1_new = raw_3.copy().load_data().apply_gradient_compensation(1)
    assert raw_1_new.compensation_grade == 1
    data_1_new, times_new = raw_1_new[:, :]
    assert_array_equal(times, times_new)
    assert (np.mean(np.abs(data_1_new - data_3)) > 1e-12)
    assert_allclose(data_1, data_1_new, **tols)
    # change back
    raw_3_new = raw_1.copy().apply_gradient_compensation(3)
    data_3_new, times_new = raw_3_new[:, :]
    assert_allclose(data_3, data_3_new, **tols)
    raw_3_new = raw_1.copy().load_data().apply_gradient_compensation(3)
    data_3_new, times_new = raw_3_new[:, :]
    assert_allclose(data_3, data_3_new, **tols)

    for load in (False, True):
        for raw in (raw_0, raw_1):
            raw_3_new = raw.copy()
            if load:
                raw_3_new.load_data()
            raw_3_new.apply_gradient_compensation(3)
            assert raw_3_new.compensation_grade == 3
            data_3_new, times_new = raw_3_new[:, :]
            assert_array_equal(times, times_new)
            assert (np.mean(np.abs(data_3_new - data_1)) > 1e-12)
            assert_allclose(data_3, data_3_new, **tols)

    # Try IO with compensation
    temp_file = tmpdir.join('raw.fif')
    raw_3.save(temp_file, overwrite=True)
    for preload in (True, False):
        raw_read = read_raw_fif(temp_file, preload=preload)
        assert raw_read.compensation_grade == 3
        data_read, times_new = raw_read[:, :]
        assert_array_equal(times, times_new)
        assert_allclose(data_3, data_read, **tols)
        raw_read.apply_gradient_compensation(1)
        data_read, times_new = raw_read[:, :]
        assert_array_equal(times, times_new)
        assert_allclose(data_1, data_read, **tols)

    # Now save the file that has modified compensation
    # and make sure the compensation is the same as it was,
    # but that we can undo it

    # These channels have norm 1e-11/1e-12, so atol=1e-18 isn't awesome,
    # but it's due to the single precision of the info['comps'] leading
    # to inexact inversions with saving/loading (casting back to single)
    # in between (e.g., 1->3->1 will degrade like this)
    looser_tols = dict(rtol=1e-6, atol=1e-18)
    raw_1.save(temp_file, overwrite=True)
    for preload in (True, False):
        raw_read = read_raw_fif(temp_file, preload=preload, verbose=True)
        assert raw_read.compensation_grade == 1
        data_read, times_new = raw_read[:, :]
        assert_array_equal(times, times_new)
        assert_allclose(data_1, data_read, **looser_tols)
        raw_read.apply_gradient_compensation(3, verbose=True)
        data_read, times_new = raw_read[:, :]
        assert_array_equal(times, times_new)
        assert_allclose(data_3, data_read, **looser_tols)


@requires_mne
def test_compensation_raw_mne(tmpdir):
    """Test Raw compensation by comparing with MNE-C."""
    def compensate_mne(fname, grad):
        tmp_fname = tmpdir.join('mne_ctf_test_raw.fif')
        cmd = ['mne_process_raw', '--raw', fname, '--save', tmp_fname,
               '--grad', str(grad), '--projoff', '--filteroff']
        run_subprocess(cmd)
        return read_raw_fif(tmp_fname, preload=True)

    for grad in [0, 2, 3]:
        raw_py = read_raw_fif(ctf_comp_fname, preload=True)
        raw_py.apply_gradient_compensation(grad)
        raw_c = compensate_mne(ctf_comp_fname, grad)
        assert_allclose(raw_py._data, raw_c._data, rtol=1e-6, atol=1e-17)
        assert raw_py.info['nchan'] == raw_c.info['nchan']
        for ch_py, ch_c in zip(raw_py.info['chs'], raw_c.info['chs']):
            for key in ('ch_name', 'coil_type', 'scanno', 'logno', 'unit',
                        'coord_frame', 'kind'):
                assert ch_py[key] == ch_c[key]
            for key in ('loc', 'unit_mul', 'range', 'cal'):
                assert_allclose(ch_py[key], ch_c[key])


@testing.requires_testing_data
def test_drop_channels_mixin():
    """Test channels-dropping functionality."""
    raw = read_raw_fif(fif_fname, preload=True)
    drop_ch = raw.ch_names[:3]
    ch_names = raw.ch_names[3:]

    ch_names_orig = raw.ch_names
    dummy = raw.copy().drop_channels(drop_ch)
    assert ch_names == dummy.ch_names
    assert ch_names_orig == raw.ch_names
    assert len(ch_names_orig) == raw._data.shape[0]

    raw.drop_channels(drop_ch)
    assert ch_names == raw.ch_names
    assert len(ch_names) == len(raw._cals)
    assert len(ch_names) == raw._data.shape[0]


@testing.requires_testing_data
def test_pick_channels_mixin():
    """Test channel-picking functionality."""
    # preload is True

    raw = read_raw_fif(fif_fname, preload=True)
    ch_names = raw.ch_names[:3]

    ch_names_orig = raw.ch_names
    dummy = raw.copy().pick_channels(ch_names)
    assert ch_names == dummy.ch_names
    assert ch_names_orig == raw.ch_names
    assert len(ch_names_orig) == raw._data.shape[0]

    raw.pick_channels(ch_names)  # copy is False
    assert ch_names == raw.ch_names
    assert len(ch_names) == len(raw._cals)
    assert len(ch_names) == raw._data.shape[0]
    pytest.raises(ValueError, raw.pick_channels, ch_names[0])

    raw = read_raw_fif(fif_fname, preload=False)
    pytest.raises(RuntimeError, raw.pick_channels, ch_names)
    pytest.raises(RuntimeError, raw.drop_channels, ch_names)


@testing.requires_testing_data
def test_equalize_channels():
    """Test equalization of channels."""
    raw1 = read_raw_fif(fif_fname, preload=True)

    raw2 = raw1.copy()
    ch_names = raw1.ch_names[2:]
    raw1.drop_channels(raw1.ch_names[:1])
    raw2.drop_channels(raw2.ch_names[1:2])
    my_comparison = [raw1, raw2]
    my_comparison = equalize_channels(my_comparison)
    for e in my_comparison:
        assert ch_names == e.ch_names


def test_memmap(tmpdir):
    """Test some interesting memmapping cases."""
    # concatenate_raw
    memmaps = [tmpdir.join(str(ii)) for ii in range(3)]
    raw_0 = read_raw_fif(test_fif_fname, preload=memmaps[0])
    assert raw_0._data.filename == memmaps[0]
    raw_1 = read_raw_fif(test_fif_fname, preload=memmaps[1])
    assert raw_1._data.filename == memmaps[1]
    raw_0.append(raw_1, preload=memmaps[2])
    assert raw_0._data.filename == memmaps[2]
    # add_channels
    orig_data = raw_0[:][0]
    new_ch_info = pick_info(raw_0.info, [0])
    new_ch_info['chs'][0]['ch_name'] = 'foo'
    new_ch_info._update_redundant()
    new_data = np.linspace(0, 1, len(raw_0.times))[np.newaxis]
    ch = RawArray(new_data, new_ch_info)
    raw_0.add_channels([ch])
    if sys.platform == 'darwin':
        assert not hasattr(raw_0._data, 'filename')
    else:
        assert raw_0._data.filename == memmaps[2]
    assert_allclose(orig_data, raw_0[:-1][0], atol=1e-7)
    assert_allclose(new_data, raw_0[-1][0], atol=1e-7)

    # now let's see if .copy() actually works; it does, but eventually
    # we should make it optionally memmap to a new filename rather than
    # create an in-memory version (filename=None)
    raw_0 = read_raw_fif(test_fif_fname, preload=memmaps[0])
    assert raw_0._data.filename == memmaps[0]
    assert raw_0._data[:1, 3:5].all()
    raw_1 = raw_0.copy()
    assert isinstance(raw_1._data, np.memmap)
    assert raw_1._data.filename is None
    raw_0._data[:] = 0.
    assert not raw_0._data.any()
    assert raw_1._data[:1, 3:5].all()
    # other things like drop_channels and crop work but do not use memmapping,
    # eventually we might want to add support for some of these as users
    # require them.


@pytest.mark.parametrize('split', (False, True))
@pytest.mark.parametrize('kind', ('file', 'bytes'))
@pytest.mark.parametrize('preload', (True, str))
def test_file_like(kind, preload, split, tmpdir):
    """Test handling with file-like objects."""
    if split:
        fname = tmpdir.join('test_raw.fif')
        read_raw_fif(test_fif_fname).save(fname, split_size='5MB')
        assert op.isfile(fname)
        assert op.isfile(str(fname)[:-4] + '-1.fif')
    else:
        fname = test_fif_fname
    if preload is str:
        preload = tmpdir.join('memmap')
    with open(str(fname), 'rb') as file_fid:
        fid = BytesIO(file_fid.read()) if kind == 'bytes' else file_fid
        assert not fid.closed
        assert not file_fid.closed
        with pytest.raises(ValueError, match='preload must be used with file'):
            read_raw_fif(fid)
        assert not fid.closed
        assert not file_fid.closed
        # Use test_preloading=False but explicitly pass the preload type
        # so that we don't bother testing preload=False
        kwargs = dict(fname=fid, preload=preload,
                      test_preloading=False, test_kwargs=False)
        if split:
            with pytest.warns(RuntimeWarning, match='Split raw file detected'):
                _test_raw_reader(read_raw_fif, **kwargs)
        else:
            _test_raw_reader(read_raw_fif, **kwargs)
        assert not fid.closed
        assert not file_fid.closed
    assert file_fid.closed


def test_str_like():
    """Test handling with str-like objects."""
    fname = pathlib.Path(test_fif_fname)
    raw_path = read_raw_fif(fname, preload=True)
    raw_str = read_raw_fif(test_fif_fname, preload=True)
    assert_allclose(raw_path._data, raw_str._data)


run_tests_if_main()
