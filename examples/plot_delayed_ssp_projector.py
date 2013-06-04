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
# View evoked response with projectors idle

times = 1e3 * epochs.times  # time in milliseconds
pl.figure()
evoked.plot()
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('MEG evoked fields (fT)')
pl.title('Magnetometers | SSP off')
pl.show()

# Now with all projectors activated.
times = evoked.times * 1e3
pl.figure()
evoked.copy().apply_proj().plot()
pl.xlim([times[0], times[-1]])
pl.xlabel('time (ms)')
pl.ylabel('MEG evoked fields (fT)')
pl.title('Magnetometers | SSP on')
pl.show()

# Finally we are going to investigate the incremental effects of the single
# projection vectors.

title = 'Incremental SSP application'
projs, evoked.info['projs'] = evoked.info['projs'], []  # pop projs
fig, axes = pl.subplots(2, 2)  # create 4 subplots for our four vectors

# As the bulk of projectors was extracted from the same source, an incremental
# 'protocol' will be informative. We could also easily visualize the impact of
# single projection vectors by deleting the vector directly after plotting
# by adding the following line to the loop:
#   evoked.del_proj(-1)
for proj, ax in zip(projs, axes.flatten()):
    evoked.add_proj(proj)  # add and apply on a copy,
    evoked.copy().apply_proj().plot(axes=ax)  # as this cannot be undone.
    ax.set_title('+ %s' % proj['desc'])
pl.suptitle(title)
pl.show()


# while this example has exposed the mechanism used for handling
# delayed SSP application we can make life easier this way:

evoked.plot(toggle_proj=True)
# Noe select / deselect the SSP projection vectors and see how the figure
# updates.
