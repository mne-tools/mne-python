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

# Note. if the reject parameter is not None but proj == False
# each epoch will be projected to inform the rejection decision.
# If, in this mode, the epoch is considered good,
# instead of the projected epochs an the original data will be included in the
# epochs object. This allows us to have both, rejection and the option to delay
# the application of our SSP projectors. This also works for preloaded data.
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12),
                    proj=False)

evoked = epochs.average()  # average epochs and get an Evoked dataset.

###############################################################################
# Interactively select / deselect the SSP projection vectors

# The toggle_proj option will open a check box that allows to reversibly select
# projection vectors. Any changes of the selection will immediately cause the
# figure to update which which is convenient for explorative purposes.

pl.figure()
evoked.plot(toggle_proj=True)
pl.show()

# However you might be interested in the underlying mechanics or you might
# want to write a systematic script that tackling the appropriate dose of SSPs.
# Here we go:

title = 'Incremental SSP application'

# let's move the proj list to another place
projs, evoked.info['projs'] = evoked.info['projs'], []
fig, axes = pl.subplots(2, 2)  # create 4 subplots for our four vectors

# As the bulk of projectors was extracted from the same source, we can simply
# iterate over our collection of projs and add them step by step to see how
# the signals change as a function of the SSPs applied. As this operation
# can't be undone we will operate on copies of the original evoked object to
# keep things reversible.

for proj, ax in zip(projs, axes.flatten()):
    evoked.add_proj(proj)  # add projs loop by loop.
    evoked.copy().apply_proj().plot(axes=ax)  # apply on a copy of evoked
    ax.set_title('+ %s' % proj['desc'])  # extract description.
pl.suptitle(title)
pl.show()

# We also could have easily visualized the impact of single projection vectors
# by deleting the vector directly after plotting
# E.g. had we appended the following line to the loop:
#   `evoked.del_proj(-1)`
