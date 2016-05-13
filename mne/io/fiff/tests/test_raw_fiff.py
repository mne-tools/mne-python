# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import glob
from copy import deepcopy
import warnings
import itertools as itt

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
from nose.tools import assert_true, assert_raises, assert_not_equal

from mne.datasets import testing
from mne.io.constants import FIFF
from mne.io import Raw, RawArray, concatenate_raws, read_raw_fif
from mne.io.tests.test_raw import _test_concat, _test_raw_reader
from mne import (concatenate_events, find_events, equalize_channels,
                 compute_proj_raw, pick_types, pick_channels, create_info)
from mne.utils import (_TempDir, requires_pandas, slow_test,
                       requires_mne, run_subprocess, run_tests_if_main)
from mne.externals.six.moves import zip, cPickle as pickle
from mne.io.proc_history import _get_sss_rank
from mne.io.pick import _picks_by_type
from mne.annotations import Annotations
from mne.tests.common import assert_naming

warnings.simplefilter('always')  # enable b/c these tests throw warnings

testing_path = testing.data_path(download=False)
data_dir = op.join(testing_path, 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')
ms_fname = op.join(testing_path, 'SSS', 'test_move_anon_raw.fif')

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
rng = np.random.RandomState(0)


def test_fix_types():
    """Test fixing of channel types
    """
    for fname, change in ((hp_fif_fname, True), (test_fif_fname, False),
                          (ctf_fname, False)):
        raw = Raw(fname)
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
            assert_true((orig_types[mag_picks] != new_types[mag_picks]).all())
            assert_true((new_types[mag_picks] ==
                         FIFF.FIFFV_COIL_VV_MAG_T3).all())


def test_concat():
    """Test RawFIF concatenation
    """
    # we trim the file to save lots of memory and some time
    tempdir = _TempDir()
    raw = read_raw_fif(test_fif_fname)
    raw.crop(0, 2., copy=False)
    test_name = op.join(tempdir, 'test_raw.fif')
    raw.save(test_name)
    # now run the standard test
    _test_concat(read_raw_fif, test_name)


@testing.requires_testing_data
def test_hash_raw():
    """Test hashing raw objects
    """
    raw = read_raw_fif(fif_fname)
    assert_raises(RuntimeError, raw.__hash__)
    raw = Raw(fif_fname).crop(0, 0.5, copy=False)
    raw.load_data()
    raw_2 = Raw(fif_fname).crop(0, 0.5, copy=False)
    raw_2.load_data()
    assert_equal(hash(raw), hash(raw_2))
    # do NOT use assert_equal here, failing output is terrible
    assert_equal(pickle.dumps(raw), pickle.dumps(raw_2))

    raw_2._data[0, 0] -= 1
    assert_not_equal(hash(raw), hash(raw_2))


@testing.requires_testing_data
def test_maxshield():
    """Test maxshield warning
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Raw(ms_fname, allow_maxshield=True)
    assert_equal(len(w), 1)
    assert_true('test_raw_fiff.py' in w[0].filename)


@testing.requires_testing_data
def test_subject_info():
    """Test reading subject information
    """
    tempdir = _TempDir()
    raw = Raw(fif_fname).crop(0, 1, copy=False)
    assert_true(raw.info['subject_info'] is None)
    # fake some subject data
    keys = ['id', 'his_id', 'last_name', 'first_name', 'birthday', 'sex',
            'hand']
    vals = [1, 'foobar', 'bar', 'foo', (1901, 2, 3), 0, 1]
    subject_info = dict()
    for key, val in zip(keys, vals):
        subject_info[key] = val
    raw.info['subject_info'] = subject_info
    out_fname = op.join(tempdir, 'test_subj_info_raw.fif')
    raw.save(out_fname, overwrite=True)
    raw_read = Raw(out_fname)
    for key in keys:
        assert_equal(subject_info[key], raw_read.info['subject_info'][key])
    assert_equal(raw.info['meas_date'], raw_read.info['meas_date'])
    raw.anonymize()
    raw.save(out_fname, overwrite=True)
    raw_read = Raw(out_fname)
    for this_raw in (raw, raw_read):
        assert_true(this_raw.info.get('subject_info') is None)
        assert_equal(this_raw.info['meas_date'], [0, 0])
    assert_equal(raw.info['file_id']['secs'], 0)
    assert_equal(raw.info['meas_id']['secs'], 0)
    # When we write out with raw.save, these get overwritten with the
    # new save time
    assert_true(raw_read.info['file_id']['secs'] > 0)
    assert_true(raw_read.info['meas_id']['secs'] > 0)


@testing.requires_testing_data
def test_copy_append():
    """Test raw copying and appending combinations
    """
    raw = Raw(fif_fname, preload=True).copy()
    raw_full = Raw(fif_fname)
    raw_full.append(raw)
    data = raw_full[:, :][0]
    assert_equal(data.shape[1], 2 * raw._data.shape[1])


@slow_test
@testing.requires_testing_data
def test_rank_estimation():
    """Test raw rank estimation
    """
    iter_tests = itt.product(
        [fif_fname, hp_fif_fname],  # sss
        ['norm', dict(mag=1e11, grad=1e9, eeg=1e5)]
    )
    for fname, scalings in iter_tests:
        raw = Raw(fname)
        (_, picks_meg), (_, picks_eeg) = _picks_by_type(raw.info,
                                                        meg_combined=True)
        n_meg = len(picks_meg)
        n_eeg = len(picks_eeg)

        raw = Raw(fname, preload=True)
        if 'proc_history' not in raw.info:
            expected_rank = n_meg + n_eeg
        else:
            mf = raw.info['proc_history'][0]['max_info']
            expected_rank = _get_sss_rank(mf) + n_eeg
        assert_array_equal(raw.estimate_rank(scalings=scalings), expected_rank)

        assert_array_equal(raw.estimate_rank(picks=picks_eeg,
                                             scalings=scalings),
                           n_eeg)

        raw = Raw(fname, preload=False)
        if 'sss' in fname:
            tstart, tstop = 0., 30.
            raw.add_proj(compute_proj_raw(raw))
            raw.apply_proj()
        else:
            tstart, tstop = 10., 20.

        raw.apply_proj()
        n_proj = len(raw.info['projs'])

        assert_array_equal(raw.estimate_rank(tstart=tstart, tstop=tstop,
                                             scalings=scalings),
                           expected_rank - (1 if 'sss' in fname else n_proj))


@testing.requires_testing_data
def test_output_formats():
    """Test saving and loading raw data using multiple formats
    """
    tempdir = _TempDir()
    formats = ['short', 'int', 'single', 'double']
    tols = [1e-4, 1e-7, 1e-7, 1e-15]

    # let's fake a raw file with different formats
    raw = Raw(test_fif_fname).crop(0, 1, copy=False)

    temp_file = op.join(tempdir, 'raw.fif')
    for ii, (fmt, tol) in enumerate(zip(formats, tols)):
        # Let's test the overwriting error throwing while we're at it
        if ii > 0:
            assert_raises(IOError, raw.save, temp_file, fmt=fmt)
        raw.save(temp_file, fmt=fmt, overwrite=True)
        raw2 = Raw(temp_file)
        raw2_data = raw2[:, :][0]
        assert_allclose(raw2_data, raw[:, :][0], rtol=tol, atol=1e-25)
        assert_equal(raw2.orig_format, fmt)


def _compare_combo(raw, new, times, n_times):
    for ti in times:  # let's do a subset of points for speed
        orig = raw[:, ti % n_times][0]
        # these are almost_equals because of possible dtype differences
        assert_allclose(orig, new[:, ti][0])


@slow_test
@testing.requires_testing_data
def test_multiple_files():
    """Test loading multiple files simultaneously
    """
    # split file
    tempdir = _TempDir()
    raw = Raw(fif_fname).crop(0, 10, copy=False)
    raw.load_data()
    raw.load_data()  # test no operation
    split_size = 3.  # in seconds
    sfreq = raw.info['sfreq']
    nsamp = (raw.last_samp - raw.first_samp)
    tmins = np.round(np.arange(0., nsamp, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp]))
    tmaxs /= sfreq
    tmins /= sfreq
    assert_equal(raw.n_times, len(raw.times))

    # going in reverse order so the last fname is the first file (need later)
    raws = [None] * len(tmins)
    for ri in range(len(tmins) - 1, -1, -1):
        fname = op.join(tempdir, 'test_raw_split-%d_raw.fif' % ri)
        raw.save(fname, tmin=tmins[ri], tmax=tmaxs[ri])
        raws[ri] = Raw(fname)
        assert_equal(len(raws[ri].times),
                     int(round((tmaxs[ri] - tmins[ri]) *
                               raw.info['sfreq'])) + 1)  # + 1 b/c inclusive
    events = [find_events(r, stim_channel='STI 014') for r in raws]
    last_samps = [r.last_samp for r in raws]
    first_samps = [r.first_samp for r in raws]

    # test concatenation of split file
    assert_raises(ValueError, concatenate_raws, raws, True, events[1:])
    all_raw_1, events1 = concatenate_raws(raws, preload=False,
                                          events_list=events)
    assert_allclose(all_raw_1.times, raw.times)
    assert_equal(raw.first_samp, all_raw_1.first_samp)
    assert_equal(raw.last_samp, all_raw_1.last_samp)
    assert_allclose(raw[:, :][0], all_raw_1[:, :][0])
    raws[0] = Raw(fname)
    all_raw_2 = concatenate_raws(raws, preload=True)
    assert_allclose(raw[:, :][0], all_raw_2[:, :][0])

    # test proper event treatment for split files
    events2 = concatenate_events(events, first_samps, last_samps)
    events3 = find_events(all_raw_2, stim_channel='STI 014')
    assert_array_equal(events1, events2)
    assert_array_equal(events1, events3)

    # test various methods of combining files
    raw = Raw(fif_fname, preload=True)
    n_times = raw.n_times
    # make sure that all our data match
    times = list(range(0, 2 * n_times, 999))
    # add potentially problematic points
    times.extend([n_times - 1, n_times, 2 * n_times - 1])

    raw_combo0 = Raw([fif_fname, fif_fname], preload=True)
    _compare_combo(raw, raw_combo0, times, n_times)
    raw_combo = Raw([fif_fname, fif_fname], preload=False)
    _compare_combo(raw, raw_combo, times, n_times)
    raw_combo = Raw([fif_fname, fif_fname], preload='memmap8.dat')
    _compare_combo(raw, raw_combo, times, n_times)
    assert_raises(ValueError, Raw, [fif_fname, ctf_fname])
    assert_raises(ValueError, Raw, [fif_fname, fif_bad_marked_fname])
    assert_equal(raw[:, :][0].shape[1] * 2, raw_combo0[:, :][0].shape[1])
    assert_equal(raw_combo0[:, :][0].shape[1], raw_combo0.n_times)

    # with all data preloaded, result should be preloaded
    raw_combo = Raw(fif_fname, preload=True)
    raw_combo.append(Raw(fif_fname, preload=True))
    assert_true(raw_combo.preload is True)
    assert_equal(raw_combo.n_times, raw_combo._data.shape[1])
    _compare_combo(raw, raw_combo, times, n_times)

    # with any data not preloaded, don't set result as preloaded
    raw_combo = concatenate_raws([Raw(fif_fname, preload=True),
                                  Raw(fif_fname, preload=False)])
    assert_true(raw_combo.preload is False)
    assert_array_equal(find_events(raw_combo, stim_channel='STI 014'),
                       find_events(raw_combo0, stim_channel='STI 014'))
    _compare_combo(raw, raw_combo, times, n_times)

    # user should be able to force data to be preloaded upon concat
    raw_combo = concatenate_raws([Raw(fif_fname, preload=False),
                                  Raw(fif_fname, preload=True)],
                                 preload=True)
    assert_true(raw_combo.preload is True)
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([Raw(fif_fname, preload=False),
                                  Raw(fif_fname, preload=True)],
                                 preload='memmap3.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([Raw(fif_fname, preload=True),
                                  Raw(fif_fname, preload=True)],
                                 preload='memmap4.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    raw_combo = concatenate_raws([Raw(fif_fname, preload=False),
                                  Raw(fif_fname, preload=False)],
                                 preload='memmap5.dat')
    _compare_combo(raw, raw_combo, times, n_times)

    # verify that combining raws with different projectors throws an exception
    raw.add_proj([], remove_existing=True)
    assert_raises(ValueError, raw.append, Raw(fif_fname, preload=True))

    # now test event treatment for concatenated raw files
    events = [find_events(raw, stim_channel='STI 014'),
              find_events(raw, stim_channel='STI 014')]
    last_samps = [raw.last_samp, raw.last_samp]
    first_samps = [raw.first_samp, raw.first_samp]
    events = concatenate_events(events, first_samps, last_samps)
    events2 = find_events(raw_combo0, stim_channel='STI 014')
    assert_array_equal(events, events2)

    # check out the len method
    assert_equal(len(raw), raw.n_times)
    assert_equal(len(raw), raw.last_samp - raw.first_samp + 1)


@testing.requires_testing_data
def test_split_files():
    """Test writing and reading of split raw files
    """
    tempdir = _TempDir()
    raw_1 = Raw(fif_fname, preload=True)
    # Test a very close corner case
    raw_crop = raw_1.copy().crop(0, 1., copy=False)

    assert_allclose(raw_1.info['buffer_size_sec'], 10., atol=1e-2)  # samp rate
    split_fname = op.join(tempdir, 'split_raw.fif')
    raw_1.save(split_fname, buffer_size_sec=1.0, split_size='10MB')

    raw_2 = Raw(split_fname)
    assert_allclose(raw_2.info['buffer_size_sec'], 1., atol=1e-2)  # samp rate
    data_1, times_1 = raw_1[:, :]
    data_2, times_2 = raw_2[:, :]
    assert_array_equal(data_1, data_2)
    assert_array_equal(times_1, times_2)

    # test the case where the silly user specifies the split files
    fnames = [split_fname]
    fnames.extend(sorted(glob.glob(op.join(tempdir, 'split_raw-*.fif'))))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        raw_2 = Raw(fnames)
    data_2, times_2 = raw_2[:, :]
    assert_array_equal(data_1, data_2)
    assert_array_equal(times_1, times_2)

    # test the case where we only end up with one buffer to write
    # (GH#3210). These tests rely on writing meas info and annotations
    # taking up a certain number of bytes, so if we change those functions
    # somehow, the numbers below for e.g. split_size might need to be
    # adjusted.
    raw_crop = raw_1.copy().crop(0, 5, copy=False)
    try:
        raw_crop.save(split_fname, split_size='1MB',  # too small a size
                      buffer_size_sec=1., overwrite=True)
    except ValueError as exp:
        assert_true('after writing measurement information' in str(exp), exp)
    try:
        raw_crop.save(split_fname,
                      split_size=3002276,  # still too small, now after Info
                      buffer_size_sec=1., overwrite=True)
    except ValueError as exp:
        assert_true('too large for the given split size' in str(exp), exp)
    # just barely big enough here; the right size to write exactly one buffer
    # at a time so we hit GH#3210 if we aren't careful
    raw_crop.save(split_fname, split_size='4.5MB',
                  buffer_size_sec=1., overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_allclose(raw_crop[:][0], raw_read[:][0], atol=1e-20)

    # Check our buffer arithmetic

    # 1 buffer required
    raw_crop = raw_1.copy().crop(0, 1, copy=False)
    raw_crop.save(split_fname, buffer_size_sec=1., overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_equal(len(raw_read._raw_extras[0]), 1)
    assert_equal(raw_read._raw_extras[0][0]['nsamp'], 301)
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    # 2 buffers required
    raw_crop.save(split_fname, buffer_size_sec=0.5, overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_equal(len(raw_read._raw_extras[0]), 2)
    assert_equal(raw_read._raw_extras[0][0]['nsamp'], 151)
    assert_equal(raw_read._raw_extras[0][1]['nsamp'], 150)
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    # 2 buffers required
    raw_crop.save(split_fname,
                  buffer_size_sec=1. - 1.01 / raw_crop.info['sfreq'],
                  overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_equal(len(raw_read._raw_extras[0]), 2)
    assert_equal(raw_read._raw_extras[0][0]['nsamp'], 300)
    assert_equal(raw_read._raw_extras[0][1]['nsamp'], 1)
    assert_allclose(raw_crop[:][0], raw_read[:][0])
    raw_crop.save(split_fname,
                  buffer_size_sec=1. - 2.01 / raw_crop.info['sfreq'],
                  overwrite=True)
    raw_read = read_raw_fif(split_fname)
    assert_equal(len(raw_read._raw_extras[0]), 2)
    assert_equal(raw_read._raw_extras[0][0]['nsamp'], 299)
    assert_equal(raw_read._raw_extras[0][1]['nsamp'], 2)
    assert_allclose(raw_crop[:][0], raw_read[:][0])


def test_load_bad_channels():
    """Test reading/writing of bad channels
    """
    tempdir = _TempDir()
    # Load correctly marked file (manually done in mne_process_raw)
    raw_marked = Raw(fif_bad_marked_fname)
    correct_bads = raw_marked.info['bads']
    raw = Raw(test_fif_fname)
    # Make sure it starts clean
    assert_array_equal(raw.info['bads'], [])

    # Test normal case
    raw.load_bad_channels(bad_file_works)
    # Write it out, read it in, and check
    raw.save(op.join(tempdir, 'foo_raw.fif'))
    raw_new = Raw(op.join(tempdir, 'foo_raw.fif'))
    assert_equal(correct_bads, raw_new.info['bads'])
    # Reset it
    raw.info['bads'] = []

    # Test bad case
    assert_raises(ValueError, raw.load_bad_channels, bad_file_wrong)

    # Test forcing the bad case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw.load_bad_channels(bad_file_wrong, force=True)
        n_found = sum(['1 bad channel' in str(ww.message) for ww in w])
        assert_equal(n_found, 1)  # there could be other irrelevant errors
        # write it out, read it in, and check
        raw.save(op.join(tempdir, 'foo_raw.fif'), overwrite=True)
        raw_new = Raw(op.join(tempdir, 'foo_raw.fif'))
        assert_equal(correct_bads, raw_new.info['bads'])

    # Check that bad channels are cleared
    raw.load_bad_channels(None)
    raw.save(op.join(tempdir, 'foo_raw.fif'), overwrite=True)
    raw_new = Raw(op.join(tempdir, 'foo_raw.fif'))
    assert_equal([], raw_new.info['bads'])


@slow_test
@testing.requires_testing_data
def test_io_raw():
    """Test IO for raw data (Neuromag + CTF + gz)
    """
    tempdir = _TempDir()
    # test unicode io
    for chars in [b'\xc3\xa4\xc3\xb6\xc3\xa9', b'a']:
        with Raw(fif_fname) as r:
            assert_true('Raw' in repr(r))
            assert_true(op.basename(fif_fname) in repr(r))
            desc1 = r.info['description'] = chars.decode('utf-8')
            temp_file = op.join(tempdir, 'raw.fif')
            r.save(temp_file, overwrite=True)
            with Raw(temp_file) as r2:
                desc2 = r2.info['description']
            assert_equal(desc1, desc2)

    # Let's construct a simple test for IO first
    raw = Raw(fif_fname).crop(0, 3.5, copy=False)
    raw.load_data()
    # put in some data that we know the values of
    data = rng.randn(raw._data.shape[0], raw._data.shape[1])
    raw._data[:, :] = data
    # save it somewhere
    fname = op.join(tempdir, 'test_copy_raw.fif')
    raw.save(fname, buffer_size_sec=1.0)
    # read it in, make sure the whole thing matches
    raw = Raw(fname)
    assert_allclose(data, raw[:, :][0], rtol=1e-6, atol=1e-20)
    # let's read portions across the 1-sec tag boundary, too
    inds = raw.time_as_index([1.75, 2.25])
    sl = slice(inds[0], inds[1])
    assert_allclose(data[:, sl], raw[:, sl][0], rtol=1e-6, atol=1e-20)

    # now let's do some real I/O
    fnames_in = [fif_fname, test_fif_gz_fname, ctf_fname]
    fnames_out = ['raw.fif', 'raw.fif.gz', 'raw.fif']
    for fname_in, fname_out in zip(fnames_in, fnames_out):
        fname_out = op.join(tempdir, fname_out)
        raw = Raw(fname_in)

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
        raw2 = Raw(fname_out)

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]
        assert_true(times2.max() <= 3)

        # Writing
        raw.save(fname_out, picks, tmin=0, tmax=5, overwrite=True)

        if fname_in == fif_fname or fname_in == fif_fname + '.gz':
            assert_equal(len(raw.info['dig']), 146)

        raw2 = Raw(fname_out)

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]

        assert_allclose(data, data2, rtol=1e-6, atol=1e-20)
        assert_allclose(times, times2)
        assert_allclose(raw.info['sfreq'], raw2.info['sfreq'], rtol=1e-5)

        # check transformations
        for trans in ['dev_head_t', 'dev_ctf_t', 'ctf_head_t']:
            if raw.info[trans] is None:
                assert_true(raw2.info[trans] is None)
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
                    assert_equal(raw_.info[trans]['from'], from_id)
                    assert_equal(raw_.info[trans]['to'], to_id)

        if fname_in == fif_fname or fname_in == fif_fname + '.gz':
            assert_allclose(raw.info['dig'][0]['r'], raw2.info['dig'][0]['r'])

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raw_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        raw.save(raw_badname)
        Raw(raw_badname)
    assert_naming(w, 'test_raw_fiff.py', 2)


@testing.requires_testing_data
def test_io_complex():
    """Test IO with complex data types
    """
    rng = np.random.RandomState(0)
    tempdir = _TempDir()
    dtypes = [np.complex64, np.complex128]

    raw = _test_raw_reader(Raw, fnames=fif_fname)
    picks = np.arange(5)
    start, stop = raw.time_as_index([0, 5])

    data_orig, _ = raw[picks, start:stop]

    for di, dtype in enumerate(dtypes):
        imag_rand = np.array(1j * rng.randn(data_orig.shape[0],
                             data_orig.shape[1]), dtype)

        raw_cp = raw.copy()
        raw_cp._data = np.array(raw_cp._data, dtype)
        raw_cp._data[picks, start:stop] += imag_rand
        # this should throw an error because it's complex
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            raw_cp.save(op.join(tempdir, 'raw.fif'), picks, tmin=0, tmax=5,
                        overwrite=True)
            # warning gets thrown on every instance b/c simplifilter('always')
            assert_equal(len(w), 1)

        raw2 = Raw(op.join(tempdir, 'raw.fif'))
        raw2_data, _ = raw2[picks, :]
        n_samp = raw2_data.shape[1]
        assert_allclose(raw2_data[:, :n_samp], raw_cp._data[picks, :n_samp])
        # with preloading
        raw2 = Raw(op.join(tempdir, 'raw.fif'), preload=True)
        raw2_data, _ = raw2[picks, :]
        n_samp = raw2_data.shape[1]
        assert_allclose(raw2_data[:, :n_samp], raw_cp._data[picks, :n_samp])


@testing.requires_testing_data
def test_getitem():
    """Test getitem/indexing of Raw
    """
    for preload in [False, True, 'memmap.dat']:
        raw = Raw(fif_fname, preload=preload)
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
        assert_array_equal(raw[-10:, :][0],
                           raw[len(raw.ch_names) - 10:, :][0])
        assert_raises(ValueError, raw.__getitem__,
                      (slice(-len(raw.ch_names) - 1), slice(None)))


@testing.requires_testing_data
def test_proj():
    """Test SSP proj operations
    """
    tempdir = _TempDir()
    for proj in [True, False]:
        raw = Raw(fif_fname, preload=False, proj=proj)
        assert_true(all(p['active'] == proj for p in raw.info['projs']))

        data, times = raw[0:2, :]
        data1, times1 = raw[0:2]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)

        # test adding / deleting proj
        if proj:
            assert_raises(ValueError, raw.add_proj, [],
                          {'remove_existing': True})
            assert_raises(ValueError, raw.del_proj, 0)
        else:
            projs = deepcopy(raw.info['projs'])
            n_proj = len(raw.info['projs'])
            raw.del_proj(0)
            assert_equal(len(raw.info['projs']), n_proj - 1)
            raw.add_proj(projs, remove_existing=False)
            # Test that already existing projections are not added.
            assert_equal(len(raw.info['projs']), n_proj)
            raw.add_proj(projs[:-1], remove_existing=True)
            assert_equal(len(raw.info['projs']), n_proj - 1)

    # test apply_proj() with and without preload
    for preload in [True, False]:
        raw = Raw(fif_fname, preload=preload, proj=False)
        data, times = raw[:, 0:2]
        raw.apply_proj()
        data_proj_1 = np.dot(raw._projector, data)

        # load the file again without proj
        raw = Raw(fif_fname, preload=preload, proj=False)

        # write the file with proj. activated, make sure proj has been applied
        raw.save(op.join(tempdir, 'raw.fif'), proj=True, overwrite=True)
        raw2 = Raw(op.join(tempdir, 'raw.fif'), proj=False)
        data_proj_2, _ = raw2[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert_true(all(p['active'] for p in raw2.info['projs']))

        # read orig file with proj. active
        raw2 = Raw(fif_fname, preload=preload, proj=True)
        data_proj_2, _ = raw2[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert_true(all(p['active'] for p in raw2.info['projs']))

        # test that apply_proj works
        raw.apply_proj()
        data_proj_2, _ = raw[:, 0:2]
        assert_allclose(data_proj_1, data_proj_2)
        assert_allclose(data_proj_2, np.dot(raw._projector, data_proj_2))

    tempdir = _TempDir()
    out_fname = op.join(tempdir, 'test_raw.fif')
    raw = read_raw_fif(test_fif_fname, preload=True).crop(0, 0.002, copy=False)
    raw.pick_types(meg=False, eeg=True)
    raw.info['projs'] = [raw.info['projs'][-1]]
    raw._data.fill(0)
    raw._data[-1] = 1.
    raw.save(out_fname)
    raw = read_raw_fif(out_fname, proj=True, preload=False)
    assert_allclose(raw[:, :][0][:1], raw[0, :][0])


@testing.requires_testing_data
def test_preload_modify():
    """Test preloading and modifying data
    """
    tempdir = _TempDir()
    for preload in [False, True, 'memmap.dat']:
        raw = Raw(fif_fname, preload=preload)

        nsamp = raw.last_samp - raw.first_samp + 1
        picks = pick_types(raw.info, meg='grad', exclude='bads')

        data = rng.randn(len(picks), nsamp // 2)

        try:
            raw[picks, :nsamp // 2] = data
        except RuntimeError as err:
            if not preload:
                continue
            else:
                raise err

        tmp_fname = op.join(tempdir, 'raw.fif')
        raw.save(tmp_fname, overwrite=True)

        raw_new = Raw(tmp_fname)
        data_new, _ = raw_new[picks, :nsamp / 2]

        assert_allclose(data, data_new)


@slow_test
@testing.requires_testing_data
def test_filter():
    """Test filtering (FIR and IIR) and Raw.apply_function interface
    """
    raw = Raw(fif_fname).crop(0, 7, copy=False)
    raw.load_data()
    sig_dec = 11
    sig_dec_notch = 12
    sig_dec_notch_fit = 12
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    filter_params = dict(picks=picks, n_jobs=2)
    raw_lp = raw.copy().filter(0., 4.0 - 0.25, **filter_params)
    raw_hp = raw.copy().filter(8.0 + 0.25, None, **filter_params)
    raw_bp = raw.copy().filter(4.0 + 0.25, 8.0 - 0.25, **filter_params)
    raw_bs = raw.copy().filter(8.0 + 0.25, 4.0 - 0.25, **filter_params)

    data, _ = raw[picks, :]

    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]

    assert_array_almost_equal(data, lp_data + bp_data + hp_data, sig_dec)
    assert_array_almost_equal(data, bp_data + bs_data, sig_dec)

    filter_params_iir = dict(picks=picks, n_jobs=2, method='iir')
    raw_lp_iir = raw.copy().filter(0., 4.0, **filter_params_iir)
    raw_hp_iir = raw.copy().filter(8.0, None, **filter_params_iir)
    raw_bp_iir = raw.copy().filter(4.0, 8.0, **filter_params_iir)
    del filter_params_iir
    lp_data_iir, _ = raw_lp_iir[picks, :]
    hp_data_iir, _ = raw_hp_iir[picks, :]
    bp_data_iir, _ = raw_bp_iir[picks, :]
    summation = lp_data_iir + hp_data_iir + bp_data_iir
    assert_array_almost_equal(data[:, 100:-100], summation[:, 100:-100],
                              sig_dec)

    # make sure we didn't touch other channels
    data, _ = raw[picks_meg[4:], :]
    bp_data, _ = raw_bp[picks_meg[4:], :]
    assert_array_equal(data, bp_data)
    bp_data_iir, _ = raw_bp_iir[picks_meg[4:], :]
    assert_array_equal(data, bp_data_iir)

    # ... and that inplace changes are inplace
    raw_copy = raw.copy()
    raw_copy.filter(None, 20., picks=picks, n_jobs=2)
    assert_true(raw._data[0, 0] != raw_copy._data[0, 0])
    assert_equal(raw.copy().filter(None, 20., **filter_params)._data,
                 raw_copy._data)

    # do a very simple check on line filtering
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        raw_bs = raw.copy().filter(60.0 + 0.5, 60.0 - 0.5, **filter_params)
        data_bs, _ = raw_bs[picks, :]
        raw_notch = raw.copy().notch_filter(
            60.0, picks=picks, n_jobs=2, method='fft')
    data_notch, _ = raw_notch[picks, :]
    assert_array_almost_equal(data_bs, data_notch, sig_dec_notch)

    # now use the sinusoidal fitting
    raw_notch = raw.copy().notch_filter(
        None, picks=picks, n_jobs=2, method='spectrum_fit')
    data_notch, _ = raw_notch[picks, :]
    data, _ = raw[picks, :]
    assert_array_almost_equal(data, data_notch, sig_dec_notch_fit)


def test_filter_picks():
    """Test filtering default channel picks"""
    ch_types = ['mag', 'grad', 'eeg', 'seeg', 'misc', 'stim', 'ecog']
    info = create_info(ch_names=ch_types, ch_types=ch_types, sfreq=256)
    raw = RawArray(data=np.zeros((len(ch_types), 1000)), info=info)

    # -- Deal with meg mag grad exception
    ch_types = ('misc', 'stim', 'meg', 'eeg', 'seeg', 'ecog')

    # -- Filter data channels
    for ch_type in ('mag', 'grad', 'eeg', 'seeg', 'ecog'):
        picks = dict((ch, ch == ch_type) for ch in ch_types)
        picks['meg'] = ch_type if ch_type in ('mag', 'grad') else False
        raw_ = raw.copy().pick_types(**picks)
        # Avoid RuntimeWarning due to Attenuation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            raw_.filter(10, 30)
        assert_true(any(['Attenuation' in str(ww.message) for ww in w]))

    # -- Error if no data channel
    for ch_type in ('misc', 'stim'):
        picks = dict((ch, ch == ch_type) for ch in ch_types)
        raw_ = raw.copy().pick_types(**picks)
        assert_raises(RuntimeError, raw_.filter, 10, 30)


@testing.requires_testing_data
def test_crop():
    """Test cropping raw files
    """
    # split a concatenated file to test a difficult case
    raw = Raw([fif_fname, fif_fname], preload=False)
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
        raws[ri] = raw.copy().crop(tmin, tmax, copy=False)
    all_raw_2 = concatenate_raws(raws, preload=False)
    assert_equal(raw.first_samp, all_raw_2.first_samp)
    assert_equal(raw.last_samp, all_raw_2.last_samp)
    assert_array_equal(raw[:, :][0], all_raw_2[:, :][0])

    tmins = np.round(np.arange(0., nsamp - 1, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp - 1]))
    tmaxs /= sfreq
    tmins /= sfreq

    # going in revere order so the last fname is the first file (need it later)
    raws = [None] * len(tmins)
    for ri, (tmin, tmax) in enumerate(zip(tmins, tmaxs)):
        raws[ri] = raw.copy().crop(tmin, tmax, copy=False)
    # test concatenation of split file
    all_raw_1 = concatenate_raws(raws, preload=False)

    all_raw_2 = raw.copy().crop(0, None, copy=False)
    for ar in [all_raw_1, all_raw_2]:
        assert_equal(raw.first_samp, ar.first_samp)
        assert_equal(raw.last_samp, ar.last_samp)
        assert_array_equal(raw[:, :][0], ar[:, :][0])

    # test shape consistency of cropped raw
    data = np.zeros((1, 1002001))
    info = create_info(1, 1000)
    raw = RawArray(data, info)
    for tmin in range(0, 1001, 100):
        raw1 = raw.copy().crop(tmin=tmin, tmax=tmin + 2, copy=False)
        assert_equal(raw1[:][0].shape, (1, 2001))


@testing.requires_testing_data
def test_resample():
    """Test resample (with I/O and multiple files)
    """
    tempdir = _TempDir()
    raw = Raw(fif_fname).crop(0, 3, copy=False)
    raw.load_data()
    raw_resamp = raw.copy()
    sfreq = raw.info['sfreq']
    # test parallel on upsample
    raw_resamp.resample(sfreq * 2, n_jobs=2, npad='auto')
    assert_equal(raw_resamp.n_times, len(raw_resamp.times))
    raw_resamp.save(op.join(tempdir, 'raw_resamp-raw.fif'))
    raw_resamp = Raw(op.join(tempdir, 'raw_resamp-raw.fif'), preload=True)
    assert_equal(sfreq, raw_resamp.info['sfreq'] / 2)
    assert_equal(raw.n_times, raw_resamp.n_times / 2)
    assert_equal(raw_resamp._data.shape[1], raw_resamp.n_times)
    assert_equal(raw._data.shape[0], raw_resamp._data.shape[0])
    # test non-parallel on downsample
    raw_resamp.resample(sfreq, n_jobs=1, npad='auto')
    assert_equal(raw_resamp.info['sfreq'], sfreq)
    assert_equal(raw._data.shape, raw_resamp._data.shape)
    assert_equal(raw.first_samp, raw_resamp.first_samp)
    assert_equal(raw.last_samp, raw.last_samp)
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
    assert_equal(raw1.first_samp, raw3.first_samp)
    assert_equal(raw1.last_samp, raw3.last_samp)
    assert_equal(raw1.info['sfreq'], raw3.info['sfreq'])

    # test resampling of stim channel

    # basic decimation
    stim = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    assert_allclose(raw.resample(8., npad='auto')._data,
                    [[1, 1, 0, 0, 1, 1, 0, 0]])

    # decimation of multiple stim channels
    raw = RawArray(2 * [stim], create_info(2, len(stim), 2 * ['stim']))
    assert_allclose(raw.resample(8., npad='auto')._data,
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw.resample(8., npad='auto')
        assert_true(len(w) == 1)

    # events are dropped in this case (warning)
    stim = [0, 1, 1, 0, 0, 1, 1, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw.resample(4., npad='auto')
        assert_true(len(w) == 1)

    # test resampling events: this should no longer give a warning
    stim = [0, 1, 1, 0, 0, 1, 1, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    events = find_events(raw)
    raw, events = raw.resample(4., events=events, npad='auto')
    assert_equal(events, np.array([[0, 0, 1], [2, 0, 1]]))

    # test copy flag
    stim = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    raw = RawArray([stim], create_info(1, len(stim), ['stim']))
    raw_resampled = raw.copy().resample(4., npad='auto')
    assert_true(raw_resampled is not raw)
    raw_resampled = raw.resample(4., npad='auto')
    assert_true(raw_resampled is raw)

    # resample should still work even when no stim channel is present
    raw = RawArray(np.random.randn(1, 100), create_info(1, 100, ['eeg']))
    raw.info['lowpass'] = 50.
    raw.resample(10, npad='auto')
    assert_equal(raw.info['lowpass'], 5.)
    assert_equal(len(raw), 10)


@testing.requires_testing_data
def test_hilbert():
    """Test computation of analytic signal using hilbert
    """
    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    raw_filt = raw.copy()
    raw_filt.filter(10, 20)
    raw_filt_2 = raw_filt.copy()

    raw2 = raw.copy()
    raw3 = raw.copy()
    raw.apply_hilbert(picks)
    raw2.apply_hilbert(picks, envelope=True, n_jobs=2)

    # Test custom n_fft
    raw_filt.apply_hilbert(picks)
    raw_filt_2.apply_hilbert(picks, n_fft=raw_filt_2.n_times + 1000)
    assert_equal(raw_filt._data.shape, raw_filt_2._data.shape)
    assert_allclose(raw_filt._data[:, 50:-50], raw_filt_2._data[:, 50:-50],
                    atol=1e-13, rtol=1e-2)
    assert_raises(ValueError, raw3.apply_hilbert, picks,
                  n_fft=raw3.n_times - 100)

    env = np.abs(raw._data[picks, :])
    assert_allclose(env, raw2._data[picks, :], rtol=1e-2, atol=1e-13)


@testing.requires_testing_data
def test_raw_copy():
    """Test Raw copy
    """
    raw = Raw(fif_fname, preload=True)
    data, _ = raw[:, :]
    copied = raw.copy()
    copied_data, _ = copied[:, :]
    assert_array_equal(data, copied_data)
    assert_equal(sorted(raw.__dict__.keys()),
                 sorted(copied.__dict__.keys()))

    raw = Raw(fif_fname, preload=False)
    data, _ = raw[:, :]
    copied = raw.copy()
    copied_data, _ = copied[:, :]
    assert_array_equal(data, copied_data)
    assert_equal(sorted(raw.__dict__.keys()),
                 sorted(copied.__dict__.keys()))


@requires_pandas
def test_to_data_frame():
    """Test raw Pandas exporter"""
    raw = Raw(test_fif_fname, preload=True)
    _, times = raw[0, :10]
    df = raw.to_data_frame()
    assert_true((df.columns == raw.ch_names).all())
    assert_array_equal(np.round(times * 1e3), df.index.values[:10])
    df = raw.to_data_frame(index=None)
    assert_true('time' in df.index.names)
    assert_array_equal(df.values[:, 0], raw._data[0] * 1e13)
    assert_array_equal(df.values[:, 2], raw._data[2] * 1e15)


def test_add_channels():
    """Test raw splitting / re-appending channel types
    """
    raw = Raw(test_fif_fname).crop(0, 1, copy=False).load_data()
    raw_nopre = Raw(test_fif_fname, preload=False)
    raw_eeg_meg = raw.copy().pick_types(meg=True, eeg=True)
    raw_eeg = raw.copy().pick_types(meg=False, eeg=True)
    raw_meg = raw.copy().pick_types(meg=True, eeg=False)
    raw_stim = raw.copy().pick_types(meg=False, eeg=False, stim=True)
    raw_new = raw_meg.copy().add_channels([raw_eeg, raw_stim])
    assert_true(
        all(ch in raw_new.ch_names
            for ch in list(raw_stim.ch_names) + list(raw_meg.ch_names))
    )
    raw_new = raw_meg.copy().add_channels([raw_eeg])

    assert_true(ch in raw_new.ch_names for ch in raw.ch_names)
    assert_array_equal(raw_new[:, :][0], raw_eeg_meg[:, :][0])
    assert_array_equal(raw_new[:, :][1], raw[:, :][1])
    assert_true(all(ch not in raw_new.ch_names for ch in raw_stim.ch_names))

    # Testing force updates
    raw_arr_info = create_info(['1', '2'], raw_meg.info['sfreq'], 'eeg')
    orig_head_t = raw_arr_info['dev_head_t']
    raw_arr = np.random.randn(2, raw_eeg.n_times)
    raw_arr = RawArray(raw_arr, raw_arr_info)
    # This should error because of conflicts in Info
    assert_raises(ValueError, raw_meg.copy().add_channels, [raw_arr])
    raw_meg.copy().add_channels([raw_arr], force_update_info=True)
    # Make sure that values didn't get overwritten
    assert_true(raw_arr.info['dev_head_t'] is orig_head_t)

    # Now test errors
    raw_badsf = raw_eeg.copy()
    raw_badsf.info['sfreq'] = 3.1415927
    raw_eeg.crop(.5, copy=False)

    assert_raises(AssertionError, raw_meg.add_channels, [raw_nopre])
    assert_raises(RuntimeError, raw_meg.add_channels, [raw_badsf])
    assert_raises(AssertionError, raw_meg.add_channels, [raw_eeg])
    assert_raises(ValueError, raw_meg.add_channels, [raw_meg])
    assert_raises(AssertionError, raw_meg.add_channels, raw_badsf)


@testing.requires_testing_data
def test_raw_time_as_index():
    """ Test time as index conversion"""
    raw = Raw(fif_fname, preload=True)
    with warnings.catch_warnings(record=True):  # deprecation
        first_samp = raw.time_as_index([0], True)[0]
    assert_equal(raw.first_samp, -first_samp)


@testing.requires_testing_data
def test_save():
    """ Test saving raw"""
    tempdir = _TempDir()
    raw = Raw(fif_fname, preload=False)
    # can't write over file being read
    assert_raises(ValueError, raw.save, fif_fname)
    raw = Raw(fif_fname, preload=True)
    # can't overwrite file without overwrite=True
    assert_raises(IOError, raw.save, fif_fname)

    # test abspath support and annotations
    annot = Annotations([10], [10], ['test'], raw.info['meas_date'])
    raw.annotations = annot
    new_fname = op.join(op.abspath(op.curdir), 'break-raw.fif')
    raw.save(op.join(tempdir, new_fname), overwrite=True)
    new_raw = Raw(op.join(tempdir, new_fname), preload=False)
    assert_raises(ValueError, new_raw.save, new_fname)
    assert_array_equal(annot.onset, new_raw.annotations.onset)
    assert_array_equal(annot.duration, new_raw.annotations.duration)
    assert_array_equal(annot.description, new_raw.annotations.description)
    assert_equal(annot.orig_time, new_raw.annotations.orig_time)
    # make sure we can overwrite the file we loaded when preload=True
    new_raw = Raw(op.join(tempdir, new_fname), preload=True)
    new_raw.save(op.join(tempdir, new_fname), overwrite=True)
    os.remove(new_fname)


@testing.requires_testing_data
def test_with_statement():
    """ Test with statement """
    for preload in [True, False]:
        with Raw(fif_fname, preload=preload) as raw_:
            print(raw_)


def test_compensation_raw():
    """Test Raw compensation
    """
    tempdir = _TempDir()
    raw1 = Raw(ctf_comp_fname, compensation=None)
    assert_true(raw1.comp is None)
    data1, times1 = raw1[:, :]
    raw2 = Raw(ctf_comp_fname, compensation=3)
    data2, times2 = raw2[:, :]
    assert_true(raw2.comp is None)  # unchanged (data come with grade 3)
    assert_array_equal(times1, times2)
    assert_array_equal(data1, data2)
    raw3 = Raw(ctf_comp_fname, compensation=1)
    data3, times3 = raw3[:, :]
    assert_true(raw3.comp is not None)
    assert_array_equal(times1, times3)
    # make sure it's different with a different compensation:
    assert_true(np.mean(np.abs(data1 - data3)) > 1e-12)
    assert_raises(ValueError, Raw, ctf_comp_fname, compensation=33)

    # Try IO with compensation
    temp_file = op.join(tempdir, 'raw.fif')

    raw1.save(temp_file, overwrite=True)
    raw4 = Raw(temp_file)
    data4, times4 = raw4[:, :]
    assert_array_equal(times1, times4)
    assert_array_equal(data1, data4)

    # Now save the file that has modified compensation
    # and make sure we can the same data as input ie. compensation
    # is undone
    raw3.save(temp_file, overwrite=True)
    raw5 = Raw(temp_file)
    data5, times5 = raw5[:, :]
    assert_array_equal(times1, times5)
    assert_allclose(data1, data5, rtol=1e-12, atol=1e-22)


@requires_mne
def test_compensation_raw_mne():
    """Test Raw compensation by comparing with MNE
    """
    tempdir = _TempDir()

    def compensate_mne(fname, grad):
        tmp_fname = op.join(tempdir, 'mne_ctf_test_raw.fif')
        cmd = ['mne_process_raw', '--raw', fname, '--save', tmp_fname,
               '--grad', str(grad), '--projoff', '--filteroff']
        run_subprocess(cmd)
        return Raw(tmp_fname, preload=True)

    for grad in [0, 2, 3]:
        raw_py = Raw(ctf_comp_fname, preload=True, compensation=grad)
        raw_c = compensate_mne(ctf_comp_fname, grad)
        assert_allclose(raw_py._data, raw_c._data, rtol=1e-6, atol=1e-17)


@testing.requires_testing_data
def test_drop_channels_mixin():
    """Test channels-dropping functionality
    """
    raw = Raw(fif_fname, preload=True)
    drop_ch = raw.ch_names[:3]
    ch_names = raw.ch_names[3:]

    ch_names_orig = raw.ch_names
    dummy = raw.copy().drop_channels(drop_ch)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, raw.ch_names)
    assert_equal(len(ch_names_orig), raw._data.shape[0])

    raw.drop_channels(drop_ch)
    assert_equal(ch_names, raw.ch_names)
    assert_equal(len(ch_names), len(raw._cals))
    assert_equal(len(ch_names), raw._data.shape[0])


@testing.requires_testing_data
def test_pick_channels_mixin():
    """Test channel-picking functionality
    """
    # preload is True

    raw = Raw(fif_fname, preload=True)
    ch_names = raw.ch_names[:3]

    ch_names_orig = raw.ch_names
    dummy = raw.copy().pick_channels(ch_names)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, raw.ch_names)
    assert_equal(len(ch_names_orig), raw._data.shape[0])

    raw.pick_channels(ch_names)  # copy is False
    assert_equal(ch_names, raw.ch_names)
    assert_equal(len(ch_names), len(raw._cals))
    assert_equal(len(ch_names), raw._data.shape[0])
    assert_raises(ValueError, raw.pick_channels, ch_names[0])

    raw = Raw(fif_fname, preload=False)
    assert_raises(RuntimeError, raw.pick_channels, ch_names)
    assert_raises(RuntimeError, raw.drop_channels, ch_names)


@testing.requires_testing_data
def test_equalize_channels():
    """Test equalization of channels
    """
    raw1 = Raw(fif_fname, preload=True)

    raw2 = raw1.copy()
    ch_names = raw1.ch_names[2:]
    raw1.drop_channels(raw1.ch_names[:1])
    raw2.drop_channels(raw2.ch_names[1:2])
    my_comparison = [raw1, raw2]
    equalize_channels(my_comparison)
    for e in my_comparison:
        assert_equal(ch_names, e.ch_names)


run_tests_if_main()
