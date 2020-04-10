"""
.. _ex-read-events:

=====================
Reading an event file
=====================

Read events from a file. For a more detailed discussion of events in
MNE-Python, see :ref:`tut-events-vs-annotations` and :ref:`tut-event-arrays`.
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Chris Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

###############################################################################
# Reading events
# --------------
#
# Below we'll read in an events file. We suggest that this file end in
# ``-eve.fif``. Note that we can read in the entire events file, or only
# events corresponding to particular event types with the ``include`` and
# ``exclude`` parameters.

events_1 = mne.read_events(fname, include=1)
events_1_2 = mne.read_events(fname, include=[1, 2])
events_not_4_32 = mne.read_events(fname, exclude=[4, 32])

###############################################################################
# Events objects are essentially numpy arrays with three columns:
# ``event_sample | previous_event_id | event_id``

print(events_1[:5], '\n\n---\n\n', events_1_2[:5], '\n\n')

for ind, before, after in events_1[:5]:
    print("At sample %d stim channel went from %d to %d"
          % (ind, before, after))

###############################################################################
# Plotting events
# ---------------
#
# We can also plot events in order to visualize how events occur over the
# course of our recording session. Below we'll plot our three event types
# to see which ones were included.

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

mne.viz.plot_events(events_1, axes=axs[0], show=False)
axs[0].set(title="restricted to event 1")

mne.viz.plot_events(events_1_2, axes=axs[1], show=False)
axs[1].set(title="restricted to event 1 or 2")

mne.viz.plot_events(events_not_4_32, axes=axs[2], show=False)
axs[2].set(title="keep all but 4 and 32")
plt.setp([ax.get_xticklabels() for ax in axs], rotation=45)
plt.tight_layout()
plt.show()

###############################################################################
# Writing events
# --------------
#
# Finally, we can write events to disk. Remember to use the naming convention
# ``-eve.fif`` for your file.

mne.write_events('example-eve.fif', events_1)
