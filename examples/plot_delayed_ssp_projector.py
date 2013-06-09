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
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne
from mne import fiff
from mne.datasets import sample
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

# pick magnetometer channels
picks = fiff.pick_types(raw.info, meg='mag', stim=False, eog=True,
                        include=[], exclude='bads')

# When we set proj to `delayed` while passing reject parameters
# each epoch will be projected in order to preserve data when performing
# the peak-to-peak amplitude rejection. If the epoch will be kept in this mode,
# the unprojected raw epoch will be included. This allows us to process our
# epochs as if we had projected them. As a consequence, the point in time at
# which the projection is applied will not affect the results, e.g., by
# imbalanced epoch counts we would get would we not have projected the epochs
# before performing the rejection. We will make use of this function to
# interactively select projections at the evoked stage.

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12),
                    proj='delayed')

evoked = epochs.average()  # average epochs and get an Evoked dataset.

###############################################################################
# Interactively select / deselect the SSP projection vectors

# Here we expose the details of how to apply SSPs reversibly
title = 'Incremental SSP application'

# let's first move the proj list to another location
projs, evoked.info['projs'] = evoked.info['projs'], []
fig, axes = pl.subplots(2, 2)  # create 4 subplots for our four vectors

# As the bulk of projectors was extracted from the same source, we can simply
# iterate over our collection of projs and add them step by step to see how
# the signals change as a function of the SSPs applied. As this operation
# can't be undone we will operate on copies of the original evoked object to
# keep things reversible.

for proj, ax in zip(projs, axes.flatten()):
    evoked.add_proj(proj)  # add projection vectors loop by loop.
    evoked.copy().apply_proj().plot(axes=ax)  # apply on a copy of evoked
    ax.set_title('+ %s' % proj['desc'])  # extract description.
pl.suptitle(title)
pl.show()

# We also could have easily visualized the impact of single projection vectors
# by deleting the vector directly after visualizing the changes.
# E.g. had we appended the following line to our loop:
#   `evoked.del_proj(-1)`

# Often, it is desirable to interactively explore things. To make this easier
# we can make use of the 'interactive' option which will open a check box that
# allows us to reversibly select projection vectors. Any changes of the
# selection will immediately cause the figure to update.

pl.figure()
evoked.plot(proj='interactive')
pl.show()
