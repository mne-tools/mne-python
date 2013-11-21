"""
=================================================
Compare evoked responses for different conditions
=================================================

In this example, an Epochs object for visual and
auditory responses is created. Both conditions
are then accessed by their respective names to
create a sensor layout plot of the related
evoked responses.

"""

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>

# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne

from mne.fiff import Raw, pick_types
from mne.viz import plot_topo
from mne.datasets import sample
data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = Raw(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
# bad channels in raw.info['bads'] will be automatically excluded

#   Set up amplitude-peak rejection values for MEG channels
reject = dict(grad=4000e-13, mag=4e-12)

# pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                   include=include, exclude='bads')

# Create epochs including different events
epochs = mne.Epochs(raw, events, dict(audio_l=1, visual_r=3), tmin, tmax,
                    picks=picks, baseline=(None, 0), reject=reject)

# Generate list of evoked objects from conditions names
evokeds = [epochs[name].average() for name in 'audio_l', 'visual_r']

###############################################################################
# Show topography for two different conditions

colors = 'yellow', 'green'
title = 'MNE sample data - left auditory and visual'

plot_topo(evokeds, color=colors, title=title)

conditions = [e.comment for e in evokeds]
for cond, col, pos in zip(conditions, colors, (0.025, 0.07)):
    pl.figtext(0.775, pos, cond, color=col, fontsize=12)

pl.show()
