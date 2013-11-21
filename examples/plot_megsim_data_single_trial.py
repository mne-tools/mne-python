"""
======================================
MEGSIM single trial simulation dataset
======================================

The MEGSIM consists of experimental and simulated MEG data
which can be useful for reproducing research results.

The MEGSIM files will be dowloaded automatically.

The datasets are documented in:
Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
(2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
Realistic Simulated and Empirical Data. Neuroinformatics 10:141-158
"""

import pylab as pl
import mne
from mne.datasets.megsim import load_data

condition = 'visual'  # or 'auditory' or 'somatosensory'

# Load experimental RAW files for the visual condition
epochs_fnames = load_data(condition=condition, data_format='single-trial',
                          data_type='simulation')

# Take only 10 trials from the same simulation setup.
epochs_fnames = [f for f in epochs_fnames if 'sim6_trial_' in f][:10]

evokeds = [mne.fiff.read_evoked(f) for f in epochs_fnames]
mean_evoked = sum(evokeds[1:], evokeds[0])

# Visualize the average
mean_evoked.plot()
