"""
====================================================================
Sensor space least squares regression
====================================================================

Predict single trial activity from a continuous variable. This example
simulates MEG data based on the auditory trials in the sample data set.
Each auditory stimulus is assumed to have a different volume; sensor
space activity is then modified based on this value. Then a single-
trial regression is performed in each sensor and timepoint individually,
resulting in an Evoked object which contains the regression coefficient
(beta value) for each combination of sensor and timepoint.

(See Hauk et al. (2006). The time course of visual word recognition as 
revealed by linear regression analysis of ERP data. Neuroimage.)
"""
# Authors: Tal Linzen <linzen@nyu.edu>
#
# License: BSD (3-clause)

print(__doc__)

import numpy as np
import scipy

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, aud_r=2)

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                        exclude='bads')

# Reject some epochs based on amplitude
reject = dict(grad=800e-13)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)

evoked = epochs.average()
evoked.comment = 'Average'

###############################################################################
# Simulate data around auditory response

# Generate a random volume attribute for all of the stimuli
mean_volume = 5
stddev_volume = 2
volume = mean_volume + np.random.randn(len(events)) * stddev_volume

# Select the subset of epochs that are in the selection (i.e. auditory stimuli) 
# and that haven't been dropped due to amplitude thresholds or other reasons
selection_volume = volume[epochs.selection]

# Assume regression coefficient is proportional to mean activity: sensor where
# the mean response to auditory stimuli is higher would presumably be more
# sensitive to the volume of the stimulus
betas = evoked.data * 2

for epoch, vol in zip(epochs, selection_volume):
    epoch += betas * vol

###############################################################################
# Run regression

def least_squares(epochs, design_matrix, names):
    n_epochs, n_channels, n_times = epochs.get_data().shape
    assert design_matrix.shape[0] == n_epochs

    # Flatten channels and timepoints into a single dimension
    y = np.reshape(epochs.get_data(), (n_epochs, n_channels * n_times))
    betas, _, _, _ = scipy.linalg.lstsq(design_matrix, y)

    beta_maps = {}
    for x, predictor in zip(betas, names):
        beta_map = evoked.copy()
        beta_map.data = np.reshape(x, (n_channels, n_times))
        beta_map.comment = predictor
        beta_maps[predictor] = beta_map
    return beta_maps

names = ['Intercept', 'Volume']

intercept = np.ones((len(epochs),))
design_matrix = np.column_stack([intercept, selection_volume])
beta_maps = least_squares(epochs, design_matrix, names)

def plot_beta_map(m, title):
    return m.plot_topomap(ch_type='grad', size=3, times=[0.1, 0.2], vmax=200)

plot_beta_map(beta_maps['Intercept'], 'Intercept')
plot_beta_map(beta_maps['Volume'], 'Volume')

# Repeat the regression with a permuted version of the predictor vector. The
# beta values should be very close to 0 this time, since the permuted volume
# values should not be correlated with neural activity
shuffled_volume = selection_volume[np.random.permutation(len(selection_volume))]
shuffled_design_matrix = np.column_stack([intercept, shuffled_volume])
shuffled_beta_maps = least_squares(epochs, shuffled_design_matrix, names)

plot_beta_map(shuffled_beta_maps['Volume'], 'Permuted volume')
