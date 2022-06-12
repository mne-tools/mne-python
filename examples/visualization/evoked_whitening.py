# -*- coding: utf-8 -*-
"""
.. _ex-evoked-whitening:

=============================================
Whitening evoked data with a noise covariance
=============================================

Evoked data are loaded and then whitened using a given noise covariance
matrix. It's an excellent quality check to see if baseline signals match
the assumption of Gaussian white noise during the baseline period.

Covariance estimation and diagnostic plots are based on
:footcite:`EngemannGramfort2015`.

References
----------
.. footbibliography::
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

# %%

import mne

from mne import io
from mne.datasets import sample
from mne.cov import compute_covariance

print(__doc__)

# %%
# Set parameters

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'

raw = io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, fir_design='firwin')
raw.info['bads'] += ['MEG 2443']  # bads + 1 more
events = mne.read_events(event_fname)

# let's look at rare events, button presses
event_id, tmin, tmax = 2, -0.2, 0.5
reject = dict(mag=4e-12, grad=4000e-13, eeg=80e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=('meg', 'eeg'),
                    baseline=None, reject=reject, preload=True)

# Uncomment next line to use fewer samples and study regularization effects
# epochs = epochs[:20]  # For your data, use as many samples as you can!

# %%
# Compute covariance using automated regularization
method_params = dict(diagonal_fixed=dict(mag=0.01, grad=0.01, eeg=0.01))
noise_covs = compute_covariance(epochs, tmin=None, tmax=0, method='auto',
                                return_estimators=True, n_jobs=None,
                                projs=None, rank=None,
                                method_params=method_params, verbose=True)

# With "return_estimator=True" all estimated covariances sorted
# by log-likelihood are returned.

print('Covariance estimates sorted from best to worst')
for c in noise_covs:
    print("%s : %s" % (c['method'], c['loglik']))

# %%
# Show the evoked data:

evoked = epochs.average()

evoked.plot(time_unit='s')  # plot evoked response

# %%
# We can then show whitening for our various noise covariance estimates.
#
# Here we should look to see if baseline signals match the
# assumption of Gaussian white noise. we expect values centered at
# 0 within 2 standard deviations for 95% of the time points.
#
# For the Global field power we expect a value of 1.

evoked.plot_white(noise_covs, time_unit='s')
