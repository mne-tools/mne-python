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
    n_rows, n_predictors = design_matrix.shape
    assert n_rows == n_epochs

    # Flatten channels and timepoints into a single dimension
    y = np.reshape(epochs.get_data(), (n_epochs, n_channels * n_times))
    betas, resid_sum_squares, _, _ = scipy.linalg.lstsq(design_matrix, y)

    df = n_rows - n_predictors
    sqrt_noise_var = np.sqrt(resid_sum_squares / df)
    sqrt_noise_var = sqrt_noise_var.reshape(n_channels, n_times)
    design_invcov = np.linalg.inv(np.dot(design_matrix.T, design_matrix))
    unscaled_stderrs = np.sqrt(np.diag(design_invcov))

    beta = {}
    stderr = {}
    t = {}
    p = {}
    for x, unscaled_stderr, pred in zip(betas, unscaled_stderrs, names):
        beta[pred] = evoked.copy()
        beta[pred].data = np.reshape(x, (n_channels, n_times))

        stderr[pred] = evoked.copy()
        stderr[pred].data = sqrt_noise_var * unscaled_stderr

        t[pred] = evoked.copy()
        t[pred].data = beta[pred].data / stderr[pred].data

        p[pred] = evoked.copy()
        cdf = scipy.stats.t.cdf(np.abs(t[pred].data), df)
        p[pred].data = (1 - cdf) * 2
    return beta, stderr, t, p

names = ['Intercept', 'Volume']

intercept = np.ones((len(epochs),))
design_matrix = np.column_stack([intercept, selection_volume])
beta, stderr, t, p = least_squares(epochs, design_matrix, names)

# Test againsts statsmodels:
def test_lstsq():
    import pandas as pd
    import statsmodels.formula.api as sm
    n_ch, n_time = t['Volume'].data.shape
    for ch in range(n_ch):
        for time in range(n_time):
            activity = epochs.get_data()[:, ch, time]
            data = pd.DataFrame({'Activity': activity,
                                 'Volume': selection_volume})
            fit = sm.ols(formula='Activity ~ Volume', data=data).fit()
            expected = fit.tvalues['Volume']
            actual = t['Volume'].data[ch, time]
            assert abs(actual - expected) < 1e-6

            expected = fit.pvalues['Volume']
            actual = p['Volume'].data[ch, time]
            assert abs(actual - expected) < 1e-6

def plot_beta_map(m):
    return m.plot_topomap(ch_type='grad', size=3, times=[0.1, 0.2], vmax=200)

plot_beta_map(beta['Intercept'])
plot_beta_map(beta['Volume'])

t['Volume'].plot_topomap(ch_type='grad', size=3, times=[0.1, 0.2], scale=1,
                         unit='t value')

signed_scaled_p = p['Volume'].copy()
a = -np.log10(p['Volume'].data) * np.sign(t['Volume'].data)
signed_scaled_p.data = a.clip(-100, 100)
signed_scaled_p.plot_topomap(ch_type='grad', size=3, times=[0.1, 0.2, 0.3],
                             scale=1, unit='-log10(p)')

# Repeat the regression with a permuted version of the predictor vector. The
# beta values should be very close to 0 this time, since the permuted volume
# values should not be correlated with neural activity
s_volume = selection_volume[np.random.permutation(len(selection_volume))]
s_design_matrix = np.column_stack([intercept, s_volume])
s_beta, s_stderr, s_t, s_p = least_squares(epochs, s_design_matrix, names)

plot_beta_map(s_beta['Volume'])
