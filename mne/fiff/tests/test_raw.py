import os.path as op
from copy import deepcopy

import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal

from .. import Raw, pick_types, pick_channels

fif_fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')
ctf_fname = op.join(op.dirname(__file__), 'data', 'test_ctf_raw.fif')


def test_io_raw():
    """Test IO for raw data (Neuromag + CTF)
    """
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
        raw = Raw(fif_fname, preload=False)
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
    raw_lp.low_pass_filter(picks, 4.0, verbose=0, n_jobs=2)

    raw_hp = deepcopy(raw)
    raw_hp.high_pass_filter(picks, 8.0, verbose=0)

    raw_bp = deepcopy(raw)
    raw_bp.band_pass_filter(picks, 4.0, 8.0, verbose=0)

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
