"""
=========================================
Create evoked objects in delayed SSP mode
=========================================

This script shows how to apply SSP projectors delayed, that is,
at the evoked stage. This is particularly useful to support decisions
related to the trade-off between denoising and preserving signal.
We first will extract Epochs and create evoked objects
with the required settings for delayed SSP application.
Then we will explore the impact of the particular SSP projectors
on the evoked data.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, method='iir')
events = mne.read_events(event_fname)

# pick magnetometer channels
picks = mne.pick_types(raw.info, meg='mag', stim=False, eog=True,
                       include=[], exclude='bads')

# If we suspend SSP projection at the epochs stage we might reject
# more epochs than necessary. To deal with this we set proj to `delayed`
# while passing reject parameters. Each epoch will then be projected before
# performing peak-to-peak amplitude rejection. If it survives the rejection
# procedure the unprojected raw epoch will be employed instead.
# As a consequence, the point in time at which the projection is applied will
# not have impact on the final results.
# We will make use of this function to prepare for interactively selecting
# projections at the evoked stage.

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, reject=dict(mag=4e-12),
                    proj='delayed')

evoked = epochs.average()  # average epochs and get an Evoked dataset.

###############################################################################
# Interactively select / deselect the SSP projection vectors

# Here we expose the details of how to apply SSPs reversibly
title = 'Incremental SSP application'

# let's first move the proj list to another location
projs, evoked.info['projs'] = evoked.info['projs'], []
fig, axes = plt.subplots(2, 2)  # create 4 subplots for our four vectors

# As the bulk of projectors was extracted from the same source, we can simply
# iterate over our collection of projs and add them step by step to see how
# the signals change as a function of the SSPs applied. As this operation
# can't be undone we will operate on copies of the original evoked object to
# keep things reversible.

for proj, ax in zip(projs, axes.flatten()):
    evoked.add_proj(proj)  # add projection vectors loop by loop.
    evoked.copy().apply_proj().plot(axes=ax)  # apply on a copy of evoked
    ax.set_title('+ %s' % proj['desc'])  # extract description.
plt.suptitle(title)
mne.viz.tight_layout()

# We also could have easily visualized the impact of single projection vectors
# by deleting the vector directly after visualizing the changes.
# E.g. had we appended the following line to our loop:
#   `evoked.del_proj(-1)`

# Often, it is desirable to interactively explore data. To make this more
# convenient we can make use of the 'interactive' option. This will open a
# check box that allows us to reversibly select projection vectors. Any
# modification of the selection will immediately cause the figure to update.

evoked.plot(proj='interactive')

# Hint: the same works with evoked.plot_topomap
