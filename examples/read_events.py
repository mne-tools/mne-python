"""
=====================
Reading an event file
=====================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

print __doc__

import mne

fname = 'MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif'

# Reading events
events = mne.read_events(fname)

# Writing events
mne.write_events('events.fif', events)

for ind, before, after in events[:5]:
    print "At sample %d stim channel went from %d to %d" % (
                                                    ind, before, after)
