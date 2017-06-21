"""
=====================================
Sensor space least squares regression
=====================================

Predict single trial activity from a continuous variable.
A single-trial regression is performed in each sensor and timepoint
individually, resulting in an Evoked object which contains the
regression coefficient (beta value) for each combination of sensor
and timepoint. Example also shows the T statistics and the associated
p-values.

Note that this example is for educational purposes and that the data used
here do not contain any significant effect.

(See Hauk et al. (2006). The time course of visual word recognition as
revealed by linear regression analysis of ERP data. Neuroimage.)
"""
# Authors: Tal Linzen <linzen@nyu.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample
from mne.stats.regression import linear_regression

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, aud_r=2)

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                       eog=False, exclude='bads')

# Reject some epochs based on amplitude
reject = dict(mag=5e-12)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)

###############################################################################
# Run regression

names = ['intercept', 'trial-count']

intercept = np.ones((len(epochs),), dtype=np.float)
design_matrix = np.column_stack([intercept,  # intercept
                                 np.linspace(0, 1, len(intercept))])

# also accepts source estimates
lm = linear_regression(epochs, design_matrix, names)


def plot_topomap(x, unit):
    x.plot_topomap(ch_type='mag', scale=1, size=1.5, vmax=np.max,
                   unit=unit, times=np.linspace(0.1, 0.2, 5))

trial_count = lm['trial-count']

plot_topomap(trial_count.beta, unit='z (beta)')
plot_topomap(trial_count.t_val, unit='t')
plot_topomap(trial_count.mlog10_p_val, unit='-log10 p')
plot_topomap(trial_count.stderr, unit='z (error)')
