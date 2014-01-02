"""
====================================================================
Sensor space regression
====================================================================
"""
# Authors: Tal Linzen <linzen@nyu.edu>
#
# License: BSD (3-clause)

print(__doc__)

import logging

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

# Assume regression coefficient is proportional to mean activity
betas = evoked.data * 0.5

for epoch, vol in zip(epochs, selection_volume):
    epoch += betas * vol

###############################################################################
# Run regression

def regress(epochs, v):
    intercept = np.ones((len(epochs),))
    design_matrix = np.column_stack([intercept, v])

    # Flatten channels and timepoints into a single dimension
    shape = epochs.get_data().shape
    y = np.reshape(epochs.get_data(), [shape[0], -1])
    betas, _, _, _ = scipy.linalg.lstsq(design_matrix, y)

    beta_maps = {}
    for x, predictor in zip(betas, ['intercept', 'volume']):
        beta_map = evoked.copy()
        beta_map.data = np.reshape(x, shape[1:])
        beta_map.comment = predictor
        beta_maps[predictor] = beta_map
    return beta_maps

beta_maps = regress(epochs, selection_volume)
plot_args = dict(ch_type='grad', size=5, times=[0.1, 0.2, 0.3])
volume_plot_args = plot_args.copy()
volume_plot_args.update(dict(vmax=50))
print 'Intercept beta map:'
beta_maps['intercept'].plot_topomap(**plot_args)
print 'Volume beta map:'
beta_maps['volume'].plot_topomap(**volume_plot_args)

# Permute predictor to make sure that beta values go down to essentially zero
shuffled = selection_volume[np.random.permutation(len(selection_volume))]
shuffled_beta_maps = regress(epochs, shuffled)
print 'Permuted volume beta map:'
p = shuffled_beta_maps['volume'].plot_topomap(**volume_plot_args)
