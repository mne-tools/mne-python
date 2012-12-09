import os.path as op
import numpy as np

from mne import fiff, read_events, Epochs
from mne.layouts import read_layout
from mne.viz import plot_topo

import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.2, 0.5

raw = fiff.Raw(raw_fname, preload=True)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                        ecg=False, eog=False)
# Use a subset of channels for plotting speed
picks = np.round(np.linspace(0, len(picks) + 1, 50)).astype(int)
epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                baseline=(None, 0))
evoked = epochs.average()


def test_plot_topo():
    """Test plotting of ERP topography
    """
    layout = read_layout('Vectorview-all')

    # Show topography
    plot_topo(evoked, layout)