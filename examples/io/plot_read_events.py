"""
=====================
Reading an event file
=====================

Read events from a file. For a more detailed guide on how to read events
using MNE-Python, see :ref:`tut_epoching_and_averaging`.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Reading events
# events = mne.read_events(fname)  # all
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

events = mne.read_events(fname, include=1)
mne.viz.plot_events(events, axes=axs[0], show=False)
axs[0].set(title="restricted to event 1")

events = mne.read_events(fname, include=[1, 2])
mne.viz.plot_events(events, axes=axs[1], show=False)
axs[1].set(title="restricted to event 1 or 2")

events = mne.read_events(fname, exclude=[4, 32])
mne.viz.plot_events(events, axes=axs[2], show=False)
axs[2].set(title="keep all but 4 and 32")
plt.setp([ax.get_xticklabels() for ax in axs], rotation=45)
plt.show()

# Writing events
mne.write_events('events.fif', events)

for ind, before, after in events[:5]:
    print("At sample %d stim channel went from %d to %d"
          % (ind, before, after))
