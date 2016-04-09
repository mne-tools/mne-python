"""
=========================
Find events in a raw file
=========================

Find events from the stimulation/trigger channel in the raw data.
The plot them to get an idea of the paradigm.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# Reading events
raw = mne.io.read_raw_fif(fname)

events = mne.find_events(raw, stim_channel='STI 014')

# Writing events
mne.write_events('sample_audvis_raw-eve.fif', events)

for ind, before, after in events[:5]:
    print("At sample %d stim channel went from %d to %d"
          % (ind, before, after))

# Plot the events to get an idea of the paradigm
# Specify colors and an event_id dictionary for the legend.
event_id = {'aud_l': 1, 'aud_r': 2, 'vis_l': 3, 'vis_r': 4, 'smiley': 5,
            'button': 32}
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=color,
                    event_id=event_id)
