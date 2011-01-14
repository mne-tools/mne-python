import os
import os.path as op

from numpy.testing import assert_array_almost_equal, assert_equal

from math import ceil
from .. import setup_read_raw, read_raw_segment_times, pick_types, \
               start_writing_raw, write_raw_buffer, finish_writing_raw


MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis_raw.fif')

def test_io_raw():
    """Test IO for raw data
    """
    raw = setup_read_raw(fname)

    nchan = raw['info']['nchan']
    ch_names = raw['info']['ch_names']
    meg_channels_idx = [k for k in range(nchan) if ch_names[k][:3]=='MEG']
    meg_channels_idx = meg_channels_idx[:5]

    data, times = read_raw_segment_times(raw, start=100, stop=115,
                                              sel=meg_channels_idx)

    # Writing

    # Set up pick list: MEG + STI 014 - bad channels
    want_meg = True
    want_eeg = False
    want_stim = False
    include = ['STI 014']

    picks = pick_types(raw['info'], meg=want_meg, eeg=want_eeg,
                            stim=want_stim, include=include,
                            exclude=raw['info']['bads'])

    print "Number of picked channels : %d" % len(picks)

    outfid, cals = start_writing_raw('raw.fif', raw['info'], picks)
    #
    #   Set up the reading parameters
    #
    start = raw['first_samp']
    stop = raw['last_samp']
    quantum_sec = 10
    quantum = int(ceil(quantum_sec * raw['info']['sfreq']))
    #
    #   Read and write all the data
    #
    for first in range(start, stop, quantum):
        last = first + quantum
        if last >= stop:
            last = stop
        try:
            data, times = raw[picks, first:last]
        except Exception as inst:
            raw['fid'].close()
            outfid.close()
            print inst

        write_raw_buffer(outfid, data, cals)
        print '[done]'

    finish_writing_raw(outfid)
    raw['fid'].close()

