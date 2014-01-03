"""
====================================================================
Source space least squares regression
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

import logging

import numpy as np
import scipy

import mne
from mne import fiff
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.source_estimate import SourceEstimate

logger = logging.getLogger('mne')
logger.setLevel(logging.WARNING)

###############################################################################
# Set parameters and read data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
inv_fname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
subjects_dir = data_path + '/subjects'

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, aud_r=2)

raw = fiff.Raw(raw_fname, preload=True)
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                        exclude='bads')

# Reject some epochs based on amplitude
reject = dict(grad=800e-13)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)

# Zoom in on a 100ms window to speed up example
epochs.crop(0.05, 0.15)

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

# Apply inverse operator
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
inverse_operator = read_inverse_operator(inv_fname)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal")

###############################################################################
# Run regression

def least_squares(stcs, design_matrix, names):
    data = np.array([x.data for x in stcs])
    n_epochs, n_vertices, n_times = data.shape
    assert design_matrix.shape[0] == n_epochs

    # Flatten channels and timepoints into a single dimension
    y = np.reshape(data, (n_epochs, n_vertices * n_times))

    betas, _, _, _ = scipy.linalg.lstsq(design_matrix, y)
    s = stcs[0]
    beta_maps = {}
    for x, predictor in zip(betas, names):
        data = np.reshape(x, (n_vertices, n_times))
        beta_map = SourceEstimate(data, vertices=s.vertno, tmin=s.tmin,
                                  tstep=s.tstep, subject=s.subject)
        beta_maps[predictor] = beta_map
    return beta_maps

names = ['Intercept', 'Volume']

intercept = np.ones((len(epochs),))
design_matrix = np.column_stack([intercept, selection_volume])
beta_maps = least_squares(stcs, design_matrix, names)

def plot_beta_map(b, fig, title):
    p = b.plot(subjects_dir=subjects_dir, figure=fig)
    p.scale_data_colormap(fmin=2, fmid=4, fmax=6, transparent=True)
    p.set_time(100)
    p.add_text(0.05, 0.95, title, title)
    return p

plot_beta_map(beta_maps['Intercept'], 0, 'Intercept')
plot_beta_map(beta_maps['Volume'], 1, 'Volume')

# Repeat the regression with a permuted version of the predictor vector. The
# beta values should be very close to 0 this time, since the permuted volume
# values should not be correlated with neural activity
shuffled_volume = selection_volume[np.random.permutation(len(selection_volume))]
shuffled_design_matrix = np.column_stack([intercept, shuffled_volume])
shuffled_beta_maps = least_squares(stcs, shuffled_design_matrix, names)

plot_beta_map(shuffled_beta_maps['Volume'], 2, 'Permuted volume')
