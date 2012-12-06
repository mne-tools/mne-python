"""
================================================
Compare Evoked Reponses for Different Conditions
================================================

In this example, epochs of different 

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

#   Set up pick list: EEG + STI 014 - bad channels (modify to your needs)
include = []  # or stim channels ['STI 014']
exclude = raw.info['bads'] + ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                                            include=include, exclude=exclude)

# Create epochs including different events
epochs = mne.Epochs(raw, events, dict(audio_l=1, visual_r=3), tmin, tmax,
                    picks=picks, baseline=(None, 0), reject=dict(eog=150e-6))

# access sub-epochs by conditions labels
epochs_au, epochs_vi = epochs['audio_l'], epochs['visual_r']

print epochs_au, epochs_au

layout = read_layout('Vectorview-all')

evoked_au, evoked_vi = epochs_au.average(), epochs_vi.average()


###############################################################################
# Show topography for two different conditions

pl.close('all')
for evoked in [evoked_au, evoked_vi]:
    pl.figure()
    title = 'MNE sample data (condition : %s)' % evoked.comment
    plot_topo(evoked, layout, title=title)
    pl.show()
