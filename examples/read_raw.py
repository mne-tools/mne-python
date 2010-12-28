"""Reading a raw file segment
"""
print __doc__

import pylab as pl
import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis_raw.fif'
# fname = 'sm02a5_raw.fif'

raw = fiff.setup_read_raw(fname)

nchan = raw['info']['nchan']
ch_names = raw['info']['ch_names']
meg_channels_idx = [k for k in range(nchan) if ch_names[k][:3]=='MEG']
meg_channels_idx = meg_channels_idx[:5]

data, times = fiff.read_raw_segment_times(raw, from_=100, to=115,
                                          sel=meg_channels_idx)

###############################################################################
# Show MEG data
pl.plot(times, data.T)
pl.xlabel('time (ms)')
pl.ylabel('MEG data (T)')
pl.show()


