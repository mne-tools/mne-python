"""
=====================
Reading an event file
=====================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Reading events
events = mne.read_events(fname)

# Writing events
mne.write_events('events.fif', events)

for ind, before, after in events[:5]:
    print "At sample %d stim channel went from %d to %d" % (
                                                    ind, before, after)
