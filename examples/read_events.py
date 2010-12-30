"""Reading an event file
"""
print __doc__

import fiff

fname = 'MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif'

event_list = fiff.read_events(fname)
fiff.write_events('events.fif', event_list)
event_list2 = fiff.read_events(fname)

from scipy import linalg
print linalg.norm(event_list - event_list2)

# ###############################################################################
# # Show MEG data
# import pylab as pl
# pl.plot(times, data.T)
# pl.xlabel('time (ms)')
# pl.ylabel('MEG data (T)')
# pl.show()
