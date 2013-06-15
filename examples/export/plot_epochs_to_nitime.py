"""
=======================
Export epochs to NiTime
=======================

This script shows how to export Epochs to the NiTime library
for further signal processing and data analysis.

"""

# Author: Denis Engemann <d.engemann@fz-juelich.de>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
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

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))

# Export to NiTime
epochs_ts = epochs.to_nitime(picks=np.arange(20), collapse=True)

###############################################################################
# Now use nitime's OO-interface to compute coherence between sensors

from nitime.analysis import MTCoherenceAnalyzer
from nitime.viz import drawmatrix_channels
import matplotlib.pylab as pl

# setup coherency analyzer
C = MTCoherenceAnalyzer(epochs_ts)

# confine analysis to 10 - 20 Hz
freq_idx = np.where((C.frequencies > 10) * (C.frequencies < 30))[0]

# compute average coherence
coh = np.mean(C.coherence[:, :, freq_idx], -1)  # Averaging on last dimension
drawmatrix_channels(coh, epochs.ch_names, color_anchor=0,
                    title='MEG gradiometer coherence')

pl.show()
