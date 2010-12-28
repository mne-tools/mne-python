"""Reading an event file
"""
print __doc__

import pylab as pl
import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif'
# fname = 'sm02a5_raw.fif'

event_list = fiff.read_events(fname)

###############################################################################
# Show MEG data
pl.plot(times, data.T)
pl.xlabel('time (ms)')
pl.ylabel('MEG data (T)')
pl.show()


