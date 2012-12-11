"""
================================================
Compare Evoked Reponses for Different Conditions
================================================

In this example, an Epochs object for visual and
auditory responses is created. Both conditions
are then accessed by their respective names to
create a sensor layout plot of the related
evoked responses.

"""

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne

from mne.fiff import Raw, pick_types
from mne.layouts import read_layout
from mne.viz import plot_topo
from mne.datasets import sample
data_path = sample.data_path('.')

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
exclude = raw.info['bads']  # bads

#   Set up amplitude-peak rejection values for MEG channels
reject = dict(grad=4000e-13, mag=4e-12)

# pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                                            include=include, exclude=exclude)

# Create epochs including different events
epochs = mne.Epochs(raw, events, dict(audio_l=1, visual_r=3), tmin, tmax,
                    picks=picks, baseline=(None, 0), reject=reject)

# Generate list of evoked objects from conditions names
evokeds = [epochs[name].average() for name in 'audio_l', 'visual_r']

###############################################################################
# Show topography for two different conditions

layout = read_layout('Vectorview-all.lout')

pl.close('all')
title = 'MNE sample data - left auditory and visual'
plot_topo(evokeds, layout, color=['y', 'g'], title=title)
pl.show()
