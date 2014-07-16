from __future__ import print_function

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import glob
from copy import deepcopy
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import (assert_true, assert_raises, assert_equal,
                        assert_not_equal)

from mne import pick_types, pick_channels
from mne.io.constants import FIFF
from mne.io import (Raw, concatenate_raws,
                    get_chpi_positions, set_eeg_reference)
from mne import concatenate_events, find_events, equalize_channels
from mne.utils import (_TempDir, requires_nitime, requires_pandas,
                       requires_mne, run_subprocess)
from mne.externals.six.moves import zip
from mne.externals.six.moves import cPickle as pickle

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'tests', 'data')
fif_fname = op.join(base_dir, 'test_raw.fif')
fif_gz_fname = op.join(base_dir, 'test_raw.fif.gz')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')
fif_bad_marked_fname = op.join(base_dir, 'test_withbads_raw.fif')
bad_file_works = op.join(base_dir, 'test_bads.txt')
bad_file_wrong = op.join(base_dir, 'test_wrong_bads.txt')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')

tempdir = _TempDir()


def test_hash_raw():
    """Test hashing raw objects
    """
    raw = Raw(fif_fname)
    assert_raises(RuntimeError, raw.__hash__)
    raw = Raw(fif_fname, preload=True).crop(0, 0.5)
    raw_2 = Raw(fif_fname, preload=True).crop(0, 0.5)
    assert_equal(hash(raw), hash(raw_2))
    # do NOT use assert_equal here, failing output is terrible
    assert_true(pickle.dumps(raw) == pickle.dumps(raw_2))

    raw_2._data[0, 0] -= 1
    assert_not_equal(hash(raw), hash(raw_2))


def test_subject_info():
    """Test reading subject information
    """
    raw = Raw(fif_fname)
    raw.crop(0, 1, False)
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
    raw_read.anonymize()
    assert_true(raw_read.info.get('subject_info') is None)
    out_fname_anon = op.join(tempdir, 'test_subj_info_anon_raw.fif')
    raw_read.save(out_fname_anon, overwrite=True)
    raw_read = Raw(out_fname_anon)
    assert_true(raw_read.info.get('subject_info') is None)


def test_get_chpi():
    """Test CHPI position computation
    """
    trans0, rot0, _ = get_chpi_positions(hp_fname)
    raw = Raw(hp_fif_fname)
    out = get_chpi_positions(raw)
    trans1, rot1, t1 = out
    trans1 = trans1[2:]
    rot1 = rot1[2:]
    # these will not be exact because they don't use equiv. time points
    assert_allclose(trans0, trans1, atol=1e-6, rtol=1e-1)
    assert_allclose(rot0, rot1, atol=1e-6, rtol=1e-1)
    # run through input checking
    assert_raises(TypeError, get_chpi_positions, 1)
    assert_raises(ValueError, get_chpi_positions, hp_fname, [1])


def test_copy_append():
    """Test raw copying and appending combinations
    """
    raw = Raw(fif_fname, preload=True).copy()
    raw_full = Raw(fif_fname)
    raw_full.append(raw)
    data = raw_full[:, :][0]
    assert_true(data.shape[1] == 2 * raw._data.shape[1])


def test_rank_estimation():
    """Test raw rank estimation
    """
    raw = Raw(fif_fname)
    picks_meg = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    n_meg = len(picks_meg)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    n_eeg = len(picks_eeg)
    raw = Raw(fif_fname, preload=True)
    assert_array_equal(raw.estimate_rank(), n_meg + n_eeg)
    assert_array_equal(raw.estimate_rank(picks=picks_eeg), n_eeg)
    raw = Raw(fif_fname, preload=False)
    raw.apply_proj()
    n_proj = len(raw.info['projs'])
    assert_array_equal(raw.estimate_rank(tstart=10, tstop=20),
                       n_meg + n_eeg - n_proj)


def test_output_formats():
    """Test saving and loading raw data using multiple formats
    """
    formats = ['short', 'int', 'single', 'double']
    tols = [1e-4, 1e-7, 1e-7, 1e-15]

    # let's fake a raw file with different formats
    raw = Raw(fif_fname, preload=True)
    raw.crop(0, 1, copy=False)

    temp_file = op.join(tempdir, 'raw.fif')
    for ii, (format, tol) in enumerate(zip(formats, tols)):
        # Let's test the overwriting error throwing while we're at it
        if ii > 0:
            assert_raises(IOError, raw.save, temp_file, format=format)
        raw.save(temp_file, format=format, overwrite=True)
        raw2 = Raw(temp_file)
        raw2_data = raw2[:, :][0]
        assert_allclose(raw2_data, raw._data, rtol=tol, atol=1e-25)
        assert_true(raw2.orig_format == format)


def _compare_combo(raw, new, times, n_times):
    for ti in times:  # let's do a subset of points for speed
        orig = raw[:, ti % n_times][0]
        # these are almost_equals because of possible dtype differences
        assert_allclose(orig, new[:, ti][0])


def test_multiple_files():
    """Test loading multiple files simultaneously
    """
    # split file
    raw = Raw(fif_fname, preload=True).crop(0, 10)
    split_size = 3.  # in seconds
    sfreq = raw.info['sfreq']
    nsamp = (raw.last_samp - raw.first_samp)
    tmins = np.round(np.arange(0., nsamp, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp]))
    tmaxs /= sfreq
    tmins /= sfreq
    assert_equal(raw.n_times, len(raw._times))

    # going in reverse order so the last fname is the first file (need later)
    raws = [None] * len(tmins)
    for ri in range(len(tmins) - 1, -1, -1):
        fname = op.join(tempdir, 'test_raw_split-%d_raw.fif' % ri)
        raw.save(fname, tmin=tmins[ri], tmax=tmaxs[ri])
        raws[ri] = Raw(fname)
    events = [find_events(r, stim_channel='STI 014') for r in raws]
    last_samps = [r.last_samp for r in raws]
    first_samps = [r.first_samp for r in raws]

    # test concatenation of split file
    assert_raises(ValueError, concatenate_raws, raws, True, events[1:])
    all_raw_1, events1 = concatenate_raws(raws, preload=False,
                                          events_list=events)
    assert_true(raw.first_samp == all_raw_1.first_samp)
    assert_true(raw.last_samp == all_raw_1.last_samp)
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
    n_times = len(raw._times)
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
    assert_true(raw[:, :][0].shape[1] * 2 == raw_combo0[:, :][0].shape[1])
    assert_true(raw_combo0[:, :][0].shape[1] == len(raw_combo0._times))

    # with all data preloaded, result should be preloaded
    raw_combo = Raw(fif_fname, preload=True)
    raw_combo.append(Raw(fif_fname, preload=True))
    assert_true(raw_combo.preload is True)
    assert_true(len(raw_combo._times) == raw_combo._data.shape[1])
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
    assert_true(len(raw) == raw.n_times)
    assert_true(len(raw) == raw.last_samp - raw.first_samp + 1)


def test_split_files():
    """Test writing and reading of split raw files
    """
    raw_1 = Raw(fif_fname, preload=True)
    split_fname = op.join(tempdir, 'split_raw.fif')
    raw_1.save(split_fname, buffer_size_sec=1.0, split_size='10MB')

    raw_2 = Raw(split_fname)
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


def test_load_bad_channels():
    """Test reading/writing of bad channels
    """
    # Load correctly marked file (manually done in mne_process_raw)
    raw_marked = Raw(fif_bad_marked_fname)
    correct_bads = raw_marked.info['bads']
    raw = Raw(fif_fname)
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


def test_io_raw():
    """Test IO for raw data (Neuromag + CTF + gz)
    """
    # test unicode io
    for chars in [b'\xc3\xa4\xc3\xb6\xc3\xa9', b'a']:
        with Raw(fif_fname) as r:
            desc1 = r.info['description'] = chars.decode('utf-8')
            temp_file = op.join(tempdir, 'raw.fif')
            r.save(temp_file, overwrite=True)
            with Raw(temp_file) as r2:
                desc2 = r2.info['description']
            assert_equal(desc1, desc2)

    # Let's construct a simple test for IO first
    raw = Raw(fif_fname, preload=True)
    raw.crop(0, 3.5)
    # put in some data that we know the values of
    data = np.random.randn(raw._data.shape[0], raw._data.shape[1])
    raw._data[:, :] = data
    # save it somewhere
    fname = op.join(tempdir, 'test_copy_raw.fif')
    raw.save(fname, buffer_size_sec=1.0)
    # read it in, make sure the whole thing matches
    raw = Raw(fname)
    assert_true(np.allclose(data, raw[:, :][0], 1e-6, 1e-20))
    # let's read portions across the 1-sec tag boundary, too
    inds = raw.time_as_index([1.75, 2.25])
    sl = slice(inds[0], inds[1])
    assert_true(np.allclose(data[:, sl], raw[:, sl][0], 1e-6, 1e-20))

    # now let's do some real I/O
    fnames_in = [fif_fname, fif_gz_fname, ctf_fname]
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
        start, stop = raw.time_as_index([0, 5])
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
        raw2 = Raw(fname_out, preload=True)

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]
        assert_true(times2.max() <= 3)

        # Writing
        raw.save(fname_out, picks, tmin=0, tmax=5, overwrite=True)

        if fname_in == fif_fname or fname_in == fif_fname + '.gz':
            assert_true(len(raw.info['dig']) == 146)

        raw2 = Raw(fname_out)

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]

        assert_true(np.allclose(data, data2, 1e-6, 1e-20))
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
                    assert_true(raw_.info[trans]['from'] == from_id)
                    assert_true(raw_.info[trans]['to'] == to_id)

        if fname_in == fif_fname or fname_in == fif_fname + '.gz':
            assert_allclose(raw.info['dig'][0]['r'], raw2.info['dig'][0]['r'])

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raw_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        raw.save(raw_badname)
        Raw(raw_badname)
    assert_true(len(w) > 0)  # len(w) should be 2 but Travis sometimes has more


def test_io_complex():
    """Test IO with complex data types
    """
    dtypes = [np.complex64, np.complex128]

    raw = Raw(fif_fname, preload=True)
    picks = np.arange(5)
    start, stop = raw.time_as_index([0, 5])

    data_orig, _ = raw[picks, start:stop]

    for di, dtype in enumerate(dtypes):
        imag_rand = np.array(1j * np.random.randn(data_orig.shape[0],
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


def test_proj():
    """Test SSP proj operations
    """
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
            assert_true(len(raw.info['projs']) == n_proj - 1)
            raw.add_proj(projs, remove_existing=False)
            assert_true(len(raw.info['projs']) == 2 * n_proj - 1)
            raw.add_proj(projs, remove_existing=True)
            assert_true(len(raw.info['projs']) == n_proj)

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


def test_preload_modify():
    """Test preloading and modifying data
    """
    for preload in [False, True, 'memmap.dat']:
        raw = Raw(fif_fname, preload=preload)

        nsamp = raw.last_samp - raw.first_samp + 1
        picks = pick_types(raw.info, meg='grad', exclude='bads')

        data = np.random.randn(len(picks), nsamp // 2)

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


def test_filter():
    """Test filtering (FIR and IIR) and Raw.apply_function interface
    """
    raw = Raw(fif_fname, preload=True).crop(0, 7, False)
    sig_dec = 11
    sig_dec_notch = 12
    sig_dec_notch_fit = 12
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    raw_lp = raw.copy()
    raw_lp.filter(0., 4.0 - 0.25, picks=picks, n_jobs=2)

    raw_hp = raw.copy()
    raw_hp.filter(8.0 + 0.25, None, picks=picks, n_jobs=2)

    raw_bp = raw.copy()
    raw_bp.filter(4.0 + 0.25, 8.0 - 0.25, picks=picks)

    raw_bs = raw.copy()
    raw_bs.filter(8.0 + 0.25, 4.0 - 0.25, picks=picks, n_jobs=2)

    data, _ = raw[picks, :]

    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]
    bs_data, _ = raw_bs[picks, :]

    assert_array_almost_equal(data, lp_data + bp_data + hp_data, sig_dec)
    assert_array_almost_equal(data, bp_data + bs_data, sig_dec)

    raw_lp_iir = raw.copy()
    raw_lp_iir.filter(0., 4.0, picks=picks, n_jobs=2, method='iir')
    raw_hp_iir = raw.copy()
    raw_hp_iir.filter(8.0, None, picks=picks, n_jobs=2, method='iir')
    raw_bp_iir = raw.copy()
    raw_bp_iir.filter(4.0, 8.0, picks=picks, method='iir')
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

    # do a very simple check on line filtering
    raw_bs = raw.copy()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        raw_bs.filter(60.0 + 0.5, 60.0 - 0.5, picks=picks, n_jobs=2)
        data_bs, _ = raw_bs[picks, :]
        raw_notch = raw.copy()
        raw_notch.notch_filter(60.0, picks=picks, n_jobs=2, method='fft')
    data_notch, _ = raw_notch[picks, :]
    assert_array_almost_equal(data_bs, data_notch, sig_dec_notch)

    # now use the sinusoidal fitting
    raw_notch = raw.copy()
    raw_notch.notch_filter(None, picks=picks, n_jobs=2, method='spectrum_fit')
    data_notch, _ = raw_notch[picks, :]
    data, _ = raw[picks, :]
    assert_array_almost_equal(data, data_notch, sig_dec_notch_fit)


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
        raws[ri] = raw.crop(tmin, tmax, True)
    all_raw_2 = concatenate_raws(raws, preload=False)
    assert_true(raw.first_samp == all_raw_2.first_samp)
    assert_true(raw.last_samp == all_raw_2.last_samp)
    assert_array_equal(raw[:, :][0], all_raw_2[:, :][0])

    tmins = np.round(np.arange(0., nsamp - 1, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp - 1]))
    tmaxs /= sfreq
    tmins /= sfreq

    # going in revere order so the last fname is the first file (need it later)
    raws = [None] * len(tmins)
    for ri, (tmin, tmax) in enumerate(zip(tmins, tmaxs)):
        raws[ri] = raw.copy()
        raws[ri].crop(tmin, tmax, False)
    # test concatenation of split file
    all_raw_1 = concatenate_raws(raws, preload=False)

    all_raw_2 = raw.crop(0, None, True)
    for ar in [all_raw_1, all_raw_2]:
        assert_true(raw.first_samp == ar.first_samp)
        assert_true(raw.last_samp == ar.last_samp)
        assert_array_equal(raw[:, :][0], ar[:, :][0])


def test_resample():
    """Test resample (with I/O and multiple files)
    """
    raw = Raw(fif_fname, preload=True).crop(0, 3, False)
    raw_resamp = raw.copy()
    sfreq = raw.info['sfreq']
    # test parallel on upsample
    raw_resamp.resample(sfreq * 2, n_jobs=2)
    assert_true(raw_resamp.n_times == len(raw_resamp._times))
    raw_resamp.save(op.join(tempdir, 'raw_resamp-raw.fif'))
    raw_resamp = Raw(op.join(tempdir, 'raw_resamp-raw.fif'), preload=True)
    assert_true(sfreq == raw_resamp.info['sfreq'] / 2)
    assert_true(raw.n_times == raw_resamp.n_times / 2)
    assert_true(raw_resamp._data.shape[1] == raw_resamp.n_times)
    assert_true(raw._data.shape[0] == raw_resamp._data.shape[0])
    # test non-parallel on downsample
    raw_resamp.resample(sfreq, n_jobs=1)
    assert_true(raw_resamp.info['sfreq'] == sfreq)
    assert_true(raw._data.shape == raw_resamp._data.shape)
    assert_true(raw.first_samp == raw_resamp.first_samp)
    assert_true(raw.last_samp == raw.last_samp)
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
    raw1.resample(10)
    raw3.resample(10)
    raw4.resample(10)
    raw3 = concatenate_raws([raw3, raw4])
    assert_array_equal(raw1._data, raw3._data)
    assert_array_equal(raw1._first_samps, raw3._first_samps)
    assert_array_equal(raw1._last_samps, raw3._last_samps)
    assert_array_equal(raw1._raw_lengths, raw3._raw_lengths)
    assert_equal(raw1.first_samp, raw3.first_samp)
    assert_equal(raw1.last_samp, raw3.last_samp)
    assert_equal(raw1.info['sfreq'], raw3.info['sfreq'])


def test_hilbert():
    """Test computation of analytic signal using hilbert
    """
    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]

    raw2 = raw.copy()
    raw.apply_hilbert(picks)
    raw2.apply_hilbert(picks, envelope=True, n_jobs=2)

    env = np.abs(raw._data[picks, :])
    assert_allclose(env, raw2._data[picks, :], rtol=1e-2, atol=1e-13)


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


@requires_nitime
def test_raw_to_nitime():
    """ Test nitime export """
    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]
    raw_ts = raw.to_nitime(picks=picks)
    assert_true(raw_ts.data.shape[0] == len(picks))

    raw = Raw(fif_fname, preload=False)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]
    raw_ts = raw.to_nitime(picks=picks)
    assert_true(raw_ts.data.shape[0] == len(picks))

    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]
    raw_ts = raw.to_nitime(picks=picks, copy=False)
    assert_true(raw_ts.data.shape[0] == len(picks))

    raw = Raw(fif_fname, preload=False)
    picks_meg = pick_types(raw.info, meg=True, exclude='bads')
    picks = picks_meg[:4]
    raw_ts = raw.to_nitime(picks=picks, copy=False)
    assert_true(raw_ts.data.shape[0] == len(picks))


@requires_pandas
def test_as_data_frame():
    """Test raw Pandas exporter"""
    raw = Raw(fif_fname, preload=True)
    df = raw.as_data_frame()
    assert_true((df.columns == raw.ch_names).all())
    df = raw.as_data_frame(use_time_index=False)
    assert_true('time' in df.columns)
    assert_array_equal(df.values[:, 1], raw._data[0] * 1e13)
    assert_array_equal(df.values[:, 3], raw._data[2] * 1e15)


def test_raw_index_as_time():
    """ Test index as time conversion"""
    raw = Raw(fif_fname, preload=True)
    t0 = raw.index_as_time([0], True)[0]
    t1 = raw.index_as_time([100], False)[0]
    t2 = raw.index_as_time([100], True)[0]
    assert_true((t2 - t1) == t0)
    # ensure we can go back and forth
    t3 = raw.index_as_time(raw.time_as_index([0], True), True)
    assert_array_almost_equal(t3, [0.0], 2)
    t3 = raw.index_as_time(raw.time_as_index(raw.info['sfreq'], True), True)
    assert_array_almost_equal(t3, [raw.info['sfreq']], 2)
    t3 = raw.index_as_time(raw.time_as_index(raw.info['sfreq'], False), False)
    assert_array_almost_equal(t3, [raw.info['sfreq']], 2)
    i0 = raw.time_as_index(raw.index_as_time([0], True), True)
    assert_true(i0[0] == 0)
    i1 = raw.time_as_index(raw.index_as_time([100], True), True)
    assert_true(i1[0] == 100)
    # Have to add small amount of time because we truncate via int casting
    i1 = raw.time_as_index(raw.index_as_time([100.0001], False), False)
    assert_true(i1[0] == 100)


def test_raw_time_as_index():
    """ Test time as index conversion"""
    raw = Raw(fif_fname, preload=True)
    first_samp = raw.time_as_index([0], True)[0]
    assert_true(raw.first_samp == -first_samp)


def test_save():
    """ Test saving raw"""
    raw = Raw(fif_fname, preload=False)
    # can't write over file being read
    assert_raises(ValueError, raw.save, fif_fname)
    raw = Raw(fif_fname, preload=True)
    # can't overwrite file without overwrite=True
    assert_raises(IOError, raw.save, fif_fname)

    # test abspath support
    new_fname = op.join(op.abspath(op.curdir), 'break-raw.fif')
    raw.save(op.join(tempdir, new_fname), overwrite=True)
    new_raw = Raw(op.join(tempdir, new_fname), preload=False)
    assert_raises(ValueError, new_raw.save, new_fname)
    # make sure we can overwrite the file we loaded when preload=True
    new_raw = Raw(op.join(tempdir, new_fname), preload=True)
    new_raw.save(op.join(tempdir, new_fname), overwrite=True)
    os.remove(new_fname)


def test_with_statement():
    """ Test with statement """
    for preload in [True, False]:
        with Raw(fif_fname, preload=preload) as raw_:
            print(raw_)


def test_compensation_raw():
    """Test Raw compensation
    """
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


def test_set_eeg_reference():
    """ Test rereference eeg data"""
    raw = Raw(fif_fname, preload=True)

    # Rereference raw data by creating a copy of original data
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'], copy=True)

    # Separate EEG channels from other channel types
    picks_eeg = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    picks_other = pick_types(raw.info, meg=True, eeg=False, eog=True,
                             stim=True, exclude='bads')

    # Get the raw EEG data and other channel data
    raw_eeg_data = raw[picks_eeg][0]
    raw_other_data = raw[picks_other][0]

    # Get the rereferenced EEG data and channel other
    reref_eeg_data = reref[picks_eeg][0]
    unref_eeg_data = reref_eeg_data + ref_data
    # Undo rereferencing of EEG channels
    reref_other_data = reref[picks_other][0]

    # Check that both EEG data and other data is the same
    assert_array_equal(raw_eeg_data, unref_eeg_data)
    assert_array_equal(raw_other_data, reref_other_data)

    # Test that data is modified in place when copy=False
    reref, ref_data = set_eeg_reference(raw, ['EEG 001', 'EEG 002'],
                                        copy=False)
    assert_true(raw is reref)


def test_drop_channels_mixin():
    """Test channels-dropping functionality
    """
    raw = Raw(fif_fname, preload=True)
    drop_ch = raw.ch_names[:3]
    ch_names = raw.ch_names[3:]

    ch_names_orig = raw.ch_names
    dummy = raw.drop_channels(drop_ch, copy=True)
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, raw.ch_names)
    assert_equal(len(ch_names_orig), raw._data.shape[0])

    raw.drop_channels(drop_ch)
    assert_equal(ch_names, raw.ch_names)
    assert_equal(len(ch_names), len(raw.cals))
    assert_equal(len(ch_names), raw._data.shape[0])


def test_pick_channels_mixin():
    """Test channel-picking functionality
    """
    # preload is True

    raw = Raw(fif_fname, preload=True)
    ch_names = raw.ch_names[:3]

    ch_names_orig = raw.ch_names
    dummy = raw.pick_channels(ch_names, copy=True)  # copy is True
    assert_equal(ch_names, dummy.ch_names)
    assert_equal(ch_names_orig, raw.ch_names)
    assert_equal(len(ch_names_orig), raw._data.shape[0])

    raw.pick_channels(ch_names, copy=False)  # copy is False
    assert_equal(ch_names, raw.ch_names)
    assert_equal(len(ch_names), len(raw.cals))
    assert_equal(len(ch_names), raw._data.shape[0])

    raw = Raw(fif_fname, preload=False)
    assert_raises(RuntimeError, raw.pick_channels, ch_names)
    assert_raises(RuntimeError, raw.drop_channels, ch_names)


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
