"""
===========================================
MEGSIM experimental and simulation datasets
===========================================

The MEGSIM consists of experimental and simulated MEG data
which can be useful for reproducing research results.

The MEGSIM files will be dowloaded automatically.

The datasets are documented in:
Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
(2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
Realistic Simulated and Empirical Data. Neuroinformatics 10:141-158
"""

import mne
from mne import find_events, Epochs, pick_types, read_evokeds
from mne.datasets.megsim import load_data

print(__doc__)

condition = 'visual'  # or 'auditory' or 'somatosensory'

# Load experimental RAW files for the visual condition
raw_fnames = load_data(condition=condition, data_format='raw',
                       data_type='experimental', verbose=True)

# Load simulation evoked files for the visual condition
evoked_fnames = load_data(condition=condition, data_format='evoked',
                          data_type='simulation', verbose=True)

raw = mne.io.read_raw_fif(raw_fnames[0])
events = find_events(raw, stim_channel="STI 014", shortest_event=1)

# Visualize raw file
raw.plot()

# Make an evoked file from the experimental data
picks = pick_types(raw.info, meg=True, eog=True, exclude='bads')

# Read epochs
event_id, tmin, tmax = 9, -0.2, 0.5
epochs = Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0),
                picks=picks, reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
evoked = epochs.average()  # average epochs and get an Evoked dataset.
evoked.plot()

# Compare to the simulated data
evoked_sim = read_evokeds(evoked_fnames[0], condition=0)
evoked_sim.plot()
