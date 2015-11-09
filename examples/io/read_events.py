"""
=====================
Reading an event file
=====================

Read events from a file.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Reading events
# events = mne.read_events(fname)  # all
events = mne.read_events(fname, include=1)  # restricted to event 1
events = mne.read_events(fname, include=[1, 2])  # restricted to event 1 or 2
events = mne.read_events(fname, exclude=[4, 32])  # keep all but 4 and 32

# Writing events
mne.write_events('events.fif', events)

for ind, before, after in events[:5]:
    print("At sample %d stim channel went from %d to %d"
          % (ind, before, after))
