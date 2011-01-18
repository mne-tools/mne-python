"""
=======================
Read and write raw data
=======================

Read and write raw data in 60-sec blocks
"""
print __doc__

from math import ceil
from mne import fiff

infile = 'MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
outfile = 'sample_audvis_small_raw.fif'

raw = fiff.setup_read_raw(infile)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False
include = ['STI 014']
# include = []
# include = ['STI101', 'STI201', 'STI301']

picks = fiff.pick_types(raw['info'], meg=want_meg, eeg=want_eeg,
                        stim=want_stim, include=include,
                        exclude=raw['info']['bads'])

print "Number of picked channels : %d" % len(picks)

outfid, cals = fiff.start_writing_raw(outfile, raw['info'], picks)
#
#   Set up the reading parameters
#
start = raw['first_samp']
stop = raw['last_samp'] + 1
quantum_sec = 10
quantum = int(ceil(quantum_sec * raw['info']['sfreq']))
#
#   To read the whole file at once set
#
# quantum     = stop - start
#
#
#   Read and write all the data
#
first_buffer = True
for first in range(start, stop, quantum):
    last = start + quantum
    if last >= stop:
        last = stop
    try:
        data, times = raw[picks, first:last]
    except Exception as inst:
        raw['fid'].close()
        outfid.close()
        print inst

    print 'Writing ... ',
    fiff.write_raw_buffer(outfid, data, cals)
    print '[done]'

fiff.finish_writing_raw(outfid)
raw['fid'].close()
