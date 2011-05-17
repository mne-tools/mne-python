import os.path as op

from numpy.testing import assert_array_almost_equal

from .. import Raw, pick_types

fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')


def test_io_raw():
    """Test IO for raw data
    """
    raw = Raw(fname)

    nchan = raw.info['nchan']
    ch_names = raw.info['ch_names']
    meg_channels_idx = [k for k in range(nchan) if ch_names[k][:3] == 'MEG']
    meg_channels_idx = meg_channels_idx[:5]

    start, stop = raw.time_to_index(0, 5)
    data, times = raw[meg_channels_idx, start:(stop+1)]

    # Set up pick list: MEG + STI 014 - bad channels
    want_meg = True
    want_eeg = False
    want_stim = False
    include = ['STI 014']

    picks = pick_types(raw.info, meg=want_meg, eeg=want_eeg,
                            stim=want_stim, include=include,
                            exclude=raw.info['bads'])
    picks = picks[:5] # take 5 first

    print "Number of picked channels : %d" % len(picks)

    # Writing
    raw.save('raw.fif', picks, tmin=0, tmax=5)

    raw2 = Raw('raw.fif')

    data2, times2 = raw2[:,:]

    assert_array_almost_equal(data, data2)
    assert_array_almost_equal(times, times2)
    assert_array_almost_equal(raw.info['dev_head_t']['trans'],
                              raw2.info['dev_head_t']['trans'])
    assert_array_almost_equal(raw.info['sfreq'], raw2.info['sfreq'])
