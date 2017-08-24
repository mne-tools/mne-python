"""
Querying epochs with rich metadata
----------------------------------

MNE allows you to include metadata along with your ``Epochs`` objects.
This is in the form of a ``pandas`` DataFrame that has one row for each
event, and an arbitrary number of columns corresponding to different
features that were collected. Columns may be of type int, float, or str.

If an Epochs object has a metadata attribute, you can select subsets
of Epochs by using pandas query syntax directly. Here we'll show
a few examples of how this looks.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import mne

# First load some data
events = mne.read_events(os.path.join(mne.datasets.sample.data_path(),
                         'MEG/sample/sample_audvis_raw-eve.fif'))
raw = mne.io.read_raw_fif(os.path.join(mne.datasets.sample.data_path(),
                          'MEG/sample/sample_audvis_raw.fif'))

# We'll create some dummy names for each event type
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}
event_id_rev = {val: key for key, val in event_id.items()}

sides, kinds = [], []
for ev in events:
    split = event_id_rev[ev[2]].lower().split('/')
    if len(split) == 2:
        kind, side = split
    else:
        kind = split[0]
        side = 'both'
    kinds.append(kind)
    sides.append(side)


# Here's a helper function we'll use later
def plot_query_results(query):
    fig = epochs[query].average().plot(show=False)
    title = fig.axes[0].get_title()
    add = 'Query: {}\nNum Epochs: {}'.format(query, len(epochs[query]))
    fig.axes[0].set(title='\n'.join([add, title]))


###############################################################################
# First we'll create our metadata object. This should be a DataFrame with each
# row corresponding to an event.

metadata = {'event_time': events[:, 0] / raw.info['sfreq'],
            'trial_number': range(len(events)),
            'kind': kinds,
            'side': sides}
metadata = pd.DataFrame(metadata)
metadata.head()

###############################################################################
# We can use this metadata object in the construction of an Epochs object. The
# object will then exist as an attribute in the newly-created epoch object.

# Here is the
epochs = mne.Epochs(raw, events, metadata=metadata, preload=True)
epochs.metadata.head()

###############################################################################
# You can select rows by passing various queries to the Epochs object. For
# example, you can select a subset of events based on the value of a column.
query = 'kind == "auditory"'
plot_query_results(query)

###############################################################################
# If a column has numeric values, you can also use numeric-style queries:
query = 'trial_number < 10'
plot_query_results(query)

###############################################################################
# It is possible to chain these queries together, giving you more expressive
# ways to select particular epochs:
query = 'trial_number < 10 and side == "left"'
plot_query_results(query)

###############################################################################
# Any query that works with ``DataFrame.query`` will work for selecting epochs.
plot_events = ['smiley', 'button']
query = 'kind in {}'.format(plot_events)
plot_query_results(query)

plt.show()
