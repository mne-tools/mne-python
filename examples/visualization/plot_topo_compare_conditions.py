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

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)


import matplotlib.pyplot as plt
import mne

from mne.viz import plot_evoked_topo
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id = 1
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

#   Set up pick list: MEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
# bad channels in raw.info['bads'] will be automatically excluded

#   Set up amplitude-peak rejection values for MEG channels
reject = dict(grad=4000e-13, mag=4e-12)

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       include=include, exclude='bads')

# Create epochs including different events
event_id = {'audio/left': 1, 'audio/right': 2,
            'visual/left': 3, 'visual/right': 4}
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0), reject=reject)

# Generate list of evoked objects from conditions names
evokeds = [epochs[name].average() for name in ('left', 'right')]

###############################################################################
# Show topography for two different conditions

colors = 'yellow', 'green'
title = 'MNE sample data - left vs right (A/V combined)'

plot_evoked_topo(evokeds, color=colors, title=title)

conditions = [e.comment for e in evokeds]
for cond, col, pos in zip(conditions, colors, (0.025, 0.07)):
    plt.figtext(0.99, pos, cond, color=col, fontsize=12,
                horizontalalignment='right')

plt.show()
