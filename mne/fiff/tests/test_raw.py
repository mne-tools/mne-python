import os.path as op

# from numpy.testing import assert_array_almost_equal, assert_equal

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

    start, stop = raw.time_to_index(45, 50) # 100 s to 115 s data segment
    data, times = raw[meg_channels_idx, start:stop]

    # Set up pick list: MEG + STI 014 - bad channels
    want_meg = True
    want_eeg = False
    want_stim = False
    include = ['STI 014']

    picks = pick_types(raw.info, meg=want_meg, eeg=want_eeg,
                            stim=want_stim, include=include,
                            exclude=raw.info['bads'])

    print "Number of picked channels : %d" % len(picks)

    # Writing
    raw.save('raw.fif', picks)
