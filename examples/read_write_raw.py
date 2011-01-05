"""Read and write raw data

Read and write raw data in 60-sec blocks
"""
print __doc__

from math import ceil
from fiff.constants import FIFF
import fiff


infile = 'MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
outfile = 'sample_audvis_small_raw.fif'

raw = fiff.setup_read_raw(infile)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False
include = ['STI 014']
# include = ['STI101', 'STI201', 'STI301']

picks = fiff.pick_types(raw['info'], meg=want_meg, eeg=want_eeg,
                        stim=want_stim, include=include,
                        exclude=raw['info']['bads'])

print "Number of picked channels : %d" % len(picks)

outfid, cals = fiff.start_writing_raw(outfile, raw['info'], picks)
#
#   Set up the reading parameters
#
from_ = raw['first_samp']
to = raw['last_samp']
quantum_sec = 10
quantum = int(ceil(quantum_sec * raw['info']['sfreq']))
#
#   To read the whole file at once set
#
# quantum     = to - from + 1;
#
#
#   Read and write all the data
#
first_buffer = True
for first in range(from_, to, quantum):
    last = first + quantum
    if last > to:
        last = to
    try:
        data, times = fiff.read_raw_segment(raw, first, last, picks)
    except Exception as inst:
        raw['fid'].close()
        outfid.close()
        print inst
    # #
    # #   You can add your own miracle here
    # #
    # print 'Writing...'
    # if first_buffer:
    #     if first > 0:
    #         fiff.write.write_int(outfid, FIFF.FIFF_FIRST_SAMPLE, first)
    #     first_buffer = False

    fiff.write_raw_buffer(outfid, data, cals)
    print '[done]'

fiff.finish_writing_raw(outfid)
raw['fid'].close()
