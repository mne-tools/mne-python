import os.path as op
from copy import deepcopy
import warnings

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal

from mne.fiff import Raw, pick_types, pick_channels, concatenate_raws
from mne.layouts import make_eeg_layout, Layout

fif_fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')
ctf_fname = op.join(op.dirname(__file__), 'data', 'test_ctf_raw.fif')
fif_bad_marked_fname = op.join(op.dirname(__file__), 'data',
                               'test_withbads_raw.fif')
bad_file_works = op.join(op.dirname(__file__), 'data', 'test_bads.txt')
bad_file_wrong = op.join(op.dirname(__file__), 'data', 'test_wrong_bads.txt')


def test_multiple_files():
    """Test loading multiple files simultaneously"""

    # split file
    raw = Raw(fif_fname, preload=True)
    split_size = 10.  # in seconds
    sfreq = raw.info['sfreq']
    nsamp = (raw.last_samp - raw.first_samp)
    tmins = np.round(np.arange(0., nsamp, split_size * sfreq))
    tmaxs = np.concatenate((tmins[1:] - 1, [nsamp]))
    tmaxs /= sfreq
    tmins /= sfreq

    # going in revere order so the last fname is the first file (need it later)
    raws = [None] * len(tmins)
    for ri in range(len(tmins) - 1, -1, -1):
        fname = 'test_raw_split-%d_raw.fif' % ri
        raw.save(fname, tmin=tmins[ri], tmax=tmaxs[ri])
        raws[ri] = Raw(fname)

    # test concatenation of split file
    all_raw_1 = concatenate_raws(raws, preload=False)
    print (raw.first_samp, all_raw_1.first_samp)
    print (raw.last_samp, all_raw_1.last_samp)
    assert_true(raw.first_samp == all_raw_1.first_samp)
    assert_true(raw.last_samp == all_raw_1.last_samp)
    assert_array_almost_equal(raw[:, :][0], all_raw_1[:, :][0])
    raws[0] = Raw(fname)
    all_raw_2 = concatenate_raws(raws, preload=True)
    assert_array_almost_equal(raw[:, :][0], all_raw_2[:, :][0])

    # test various methods of combining files
    n_combos = 9
    raw_combos = [None] * n_combos

    raw = Raw(fif_fname, preload=True)
    raw_combos[0] = Raw([fif_fname, fif_fname], preload=True)
    raw_combos[1] = Raw([fif_fname, fif_fname], preload=False)
    raw_combos[2] = Raw([fif_fname, fif_fname], preload='memmap8.dat')
    assert_raises(ValueError, Raw, [fif_fname, ctf_fname])
    assert_raises(ValueError, Raw, [fif_fname, fif_bad_marked_fname])
    n_times = len(raw._times)
    assert_true(raw[:, :][0].shape[1] * 2 == raw_combos[0][:, :][0].shape[1])
    assert_true(raw_combos[0][:, :][0].shape[1] == len(raw_combos[0]._times))

    # with all data preloaded, result should be preloaded
    raw_combos[3] = Raw(fif_fname, preload=True)
    raw_combos[3].append(Raw(fif_fname, preload=True))
    assert_true(raw_combos[0]._preloaded == True)

    # with any data not preloaded, don't set result as preloaded
    raw_combos[4] = concatenate_raws([Raw(fif_fname, preload=True),
                                      Raw(fif_fname, preload=False)])
    assert_true(raw_combos[1]._preloaded == False)

    # user should be able to force data to be preloaded upon concat
    raw_combos[5] = concatenate_raws([Raw(fif_fname, preload=False),
                                      Raw(fif_fname, preload=True)],
                                      preload=True)
    assert_true(raw_combos[2]._preloaded == True)

    raw_combos[6] = concatenate_raws([Raw(fif_fname, preload=False),
                                      Raw(fif_fname, preload=True)],
                                      preload='memmap3.dat')

    raw_combos[7] = concatenate_raws([Raw(fif_fname, preload=True),
                                      Raw(fif_fname, preload=True)],
                                      preload='memmap4.dat')

    raw_combos[8] = concatenate_raws([Raw(fif_fname, preload=False),
                                      Raw(fif_fname, preload=False)],
                                      preload='memmap5.dat')

    # make sure that all our data match
    times = range(0, 2 * n_times, 999)
    # add potentially problematic points
    times.extend([n_times - 1, n_times, 2 * n_times - 1])
    for ti in times:  # let's do a subset of points for speed
        orig = raw[:, ti % n_times][0]
        for raw_combo in raw_combos:
            # these are almost_equals because of possible dtype differences
            assert_array_almost_equal(orig, raw_combo[:, ti][0])

    # deal with different projectors
    raw.add_proj([], remove_existing=True)
    # this shoud append, but the projectors shouldn't match
    raw.append(Raw(fif_fname, preload=True))
    # which means it should throw an error here
    assert_raises(RuntimeError, raw.apply_projector)
    # and here
    assert_raises(ValueError, raw.add_proj, [])


def test_load_bad_channels():
    """ Test reading/writing of bad channels """

    # Load correctly marked file (manually done in mne_process_raw)
    raw_marked = Raw(fif_bad_marked_fname)
    correct_bads = raw_marked.info['bads']
    raw = Raw(fif_fname)
    # Make sure it starts clean
    assert_array_equal(raw.info['bads'], [])

    # Test normal case
    raw.load_bad_channels(bad_file_works)
    # Write it out, read it in, and check
    raw.save('foo_raw.fif')
    raw_new = Raw('foo_raw.fif')
    assert_equal(correct_bads, raw_new.info['bads'])
    # Reset it
    raw.info['bads'] = []

    # Test bad case
    assert_raises(ValueError, raw.load_bad_channels, bad_file_wrong)

    # Test forcing the bad case
    with warnings.catch_warnings(record=True) as w:
        raw.load_bad_channels(bad_file_wrong, force=True)
        assert_equal(len(w), 1)
        # write it out, read it in, and check
        raw.save('foo_raw.fif')
        raw_new = Raw('foo_raw.fif')
        assert_equal(correct_bads, raw_new.info['bads'])

    # Check that bad channels are cleared
    raw.load_bad_channels(None)
    raw.save('foo_raw.fif')
    raw_new = Raw('foo_raw.fif')
    assert_equal([], raw_new.info['bads'])


def test_io_raw():
    """Test IO for raw data (Neuromag + CTF)"""
    for fname in [fif_fname, ctf_fname]:
        raw = Raw(fname)

        nchan = raw.info['nchan']
        ch_names = raw.info['ch_names']
        meg_channels_idx = [k for k in range(nchan)
                                            if ch_names[k][0] == 'M']
        n_channels = 100
        meg_channels_idx = meg_channels_idx[:n_channels]
        start, stop = raw.time_to_index(0, 5)
        data, times = raw[meg_channels_idx, start:(stop + 1)]
        meg_ch_names = [ch_names[k] for k in meg_channels_idx]

        # Set up pick list: MEG + STI 014 - bad channels
        include = ['STI 014']
        include += meg_ch_names
        picks = pick_types(raw.info, meg=True, eeg=False,
                                stim=True, misc=True, include=include,
                                exclude=raw.info['bads'])
        print "Number of picked channels : %d" % len(picks)

        # Writing with drop_small_buffer True
        raw.save('raw.fif', picks, tmin=0, tmax=4, buffer_size_sec=3,
                 drop_small_buffer=True)
        raw2 = Raw('raw.fif')

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]
        assert_true(times2.max() <= 3)

        # Writing
        raw.save('raw.fif', picks, tmin=0, tmax=5)

        if fname == fif_fname:
            assert_true(len(raw.info['dig']) == 146)

        raw2 = Raw('raw.fif')

        sel = pick_channels(raw2.ch_names, meg_ch_names)
        data2, times2 = raw2[sel, :]

        assert_array_almost_equal(data, data2)
        assert_array_almost_equal(times, times2)
        assert_array_almost_equal(raw.info['dev_head_t']['trans'],
                                  raw2.info['dev_head_t']['trans'])
        assert_array_almost_equal(raw.info['sfreq'], raw2.info['sfreq'])

        if fname == fif_fname:
            assert_array_almost_equal(raw.info['dig'][0]['r'],
                                      raw2.info['dig'][0]['r'])

        fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')


def test_io_complex():
    """ Test IO with complex data types """
    dtypes = [np.complex64, np.complex128]

    raw = Raw(fif_fname, preload=True)
    picks = np.arange(5)
    start, stop = raw.time_to_index(0, 5)

    data_orig, _ = raw[picks, start:stop]

    for dtype in dtypes:
        imag_rand = np.array(1j * np.random.randn(data_orig.shape[0],
                            data_orig.shape[1]), dtype)

        raw_cp = deepcopy(raw)
        raw_cp._data = np.array(raw_cp._data, dtype)
        raw_cp._data[picks, start:stop] += imag_rand
        raw_cp.save('raw.fif', picks, tmin=0, tmax=5)

        raw2 = Raw('raw.fif')
        raw2_data, _ = raw2[picks, :]
        n_samp = raw2_data.shape[1]
        assert_array_almost_equal(raw2_data[:, :n_samp],
                                  raw_cp._data[picks, :n_samp])


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
    """Test getitem with and without proj
    """
    for proj in [True, False]:
        raw = Raw(fif_fname, preload=False, proj=proj)
        data, times = raw[0:2, :]
        data1, times1 = raw[0:2]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)

        projs = raw.info['projs']
        raw.info['projs'] = []
        raw.add_proj(projs)
        data1, times1 = raw[[0, 1]]
        assert_array_equal(data, data1)
        assert_array_equal(times, times1)

    # test apply_proj() with and without preload
    for preload in [True, False]:
        raw = Raw(fif_fname, preload=preload, proj=False)
        # Use all sensors and a couple time points so projection works
        data, times = raw[:, 0:2]
        raw.apply_projector()
        projector = raw._projectors[0]
        data_proj_1 = np.dot(projector, data)
        data_proj_2, _ = raw[:, 0:2]
        assert_array_almost_equal(data_proj_1, data_proj_2)
        assert_array_almost_equal(data_proj_2, np.dot(projector, data_proj_2))


def test_preload_modify():
    """ Test preloading and modifying data
    """
    for preload in [False, True, 'memmap.dat']:
        raw = Raw(fif_fname, preload=preload)

        nsamp = raw.last_samp - raw.first_samp + 1
        picks = pick_types(raw.info, meg='grad')

        data = np.random.randn(len(picks), nsamp / 2)

        try:
            raw[picks, :nsamp / 2] = data
        except RuntimeError as err:
            if not preload:
                continue
            else:
                raise err

        tmp_fname = 'raw.fif'
        raw.save(tmp_fname)

        raw_new = Raw(tmp_fname)
        data_new, _ = raw_new[picks, :nsamp / 2]

        assert_array_almost_equal(data, data_new)


def test_filter():
    """ Test filtering and Raw.apply_function interface """

    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True)
    picks = picks_meg[:4]

    raw_lp = deepcopy(raw)
    raw_lp.filter(0., 4.0, picks=picks, verbose=0, n_jobs=2)

    raw_hp = deepcopy(raw)
    raw_lp.filter(8.0, None, picks=picks, verbose=0, n_jobs=2)

    raw_bp = deepcopy(raw)
    raw_bp.filter(4.0, 8.0, picks=picks, verbose=0)

    data, _ = raw[picks, :]

    lp_data, _ = raw_lp[picks, :]
    hp_data, _ = raw_hp[picks, :]
    bp_data, _ = raw_bp[picks, :]

    assert_array_almost_equal(data, lp_data + hp_data + bp_data)

    # make sure we didn't touch other channels
    data, _ = raw[picks_meg[4:], :]
    bp_data, _ = raw_bp[picks_meg[4:], :]

    assert_array_equal(data, bp_data)


def test_hilbert():
    """ Test computation of analytic signal using hilbert """
    raw = Raw(fif_fname, preload=True)
    picks_meg = pick_types(raw.info, meg=True)
    picks = picks_meg[:4]

    raw2 = deepcopy(raw)
    raw.apply_hilbert(picks, verbose=0)
    raw2.apply_hilbert(picks, envelope=True, n_jobs=2, verbose=0)

    env = np.abs(raw._data[picks, :])
    assert_array_almost_equal(env, raw2._data[picks, :])
