"""
=============================================
Whitening evoked data with a noise covariance
=============================================

Evoked data are loaded and then whitened using a given noise covariance
matrix. It's an excellent quality check to see if baseline signals match
the assumption of Gaussian white noise from which we expect values around
0 with less than 2 standard deviations. Covariance estimation and diagnostic
plots are based on [1].

References
----------
[1] Engemann D. and Gramfort A. Automated model selection in covariance
    estimation and spatial whitening of MEG and EEG signals. (in press.)
    NeuroImage.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import mne

from mne import io
from mne.datasets import sample
from mne.cov import compute_covariance

###############################################################################
# Set parameters

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

raw = io.Raw(raw_fname, preload=True)
raw.filter(1, 30, method='iir', n_jobs=4)
raw.info['bads'] += ['MEG 2443']  # bads + 1 more
events = mne.read_events(event_fname)

# let's look at rare events, button presses
event_id, tmin, tmax = 1, -0.2, 0.5
picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
reject = dict(mag=4e-12, grad=4000e-13, eeg=80e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, reject=reject, preload=True)

# Uncomment next line to use fewer samples and study regularization effects
# epochs = epochs[:20]  # For your data, use as many samples as you can!

###############################################################################
# Compute covariance using automated regularization

# the best estimator in this list will be selected
method = ('empirical', 'shrunk', 'ledoit_wolf', 'factor_analysis')
noise_covs = compute_covariance(epochs, tmin=None, tmax=0, method=method,
                                return_estimators=True, verbose=True, n_jobs=1)

# With "return_estimator=True" all estimated covariances sorted
# by log-likelihood are returned.

print('Covariance estimates sorted from best to worst')
for c in noise_covs:
    print("%s : %s" % (c['method'], c['loglik']))

###############################################################################
# Show whitening

evoked = epochs.average()

evoked.plot()  # plot evoked response

# plot the whitened evoked data for to see if baseline signals match the
# assumption of Gaussian white noise from which we expect values around
# 0 with less than 2 standard deviations. For the Global field power we expect
# a value of 1.

evoked.plot_white(noise_covs)
