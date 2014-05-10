"""
=========================
Find events in a raw file
=========================

Find events from the stimulation/trigger channel in the raw data.
The plot them to get an idea of the paradigm.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
from mne.io import Raw

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Reading events
raw = Raw(fname)

events = mne.find_events(raw, stim_channel='STI 014')

# Writing events
mne.write_events('events.fif', events)

for ind, before, after in events[:5]:
    print("At sample %d stim channel went from %d to %d"
          % (ind, before, after))

# Plot the events to get an idea of the paradigm
mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
