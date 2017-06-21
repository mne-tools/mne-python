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

from mne import read_evokeds, combine_evoked
from mne.datasets.megsim import load_data

print(__doc__)

condition = 'visual'  # or 'auditory' or 'somatosensory'

# Load experimental RAW files for the visual condition
epochs_fnames = load_data(condition=condition, data_format='single-trial',
                          data_type='simulation', verbose=True)

# Take only 10 trials from the same simulation setup.
epochs_fnames = [f for f in epochs_fnames if 'sim6_trial_' in f][:10]

evokeds = [read_evokeds(f)[0] for f in epochs_fnames]
mean_evoked = combine_evoked(evokeds, weights='nave')

# Visualize the average
mean_evoked.plot()
