import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

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
            assert_array_almost_equal(raw.info['dig'][0]['r'], raw2.info['dig'][0]['r'])

        fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')
