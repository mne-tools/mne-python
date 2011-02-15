"""
=======================
Read and write raw data
=======================

Read and write raw data in 60-sec blocks
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
from math import ceil
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')
infile = data_path
infile += '/MEG/sample/sample_audvis_raw.fif'
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
    last = first + quantum
    if last >= stop:
        last = stop+1

    data, times = raw[picks, first:last]

    print 'Writing ... ',
    fiff.write_raw_buffer(outfid, data, cals)
    print '[done]'

fiff.finish_writing_raw(outfid)
raw['fid'].close()
