"""
=============================================
Whitening evoked data with a noise covariance
=============================================

Evoked data are loaded and then whitened using a given noise covariance
matrix. It's an excellent quality check to see if baseline signals match
the assumption of Gaussian white noise from which we expect values around
and less than 2 standard deviations. Covariance estimation and diagnostic
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import mne
from mne import io
from mne.datasets import sample
from mne.cov import compute_covariance, whiten_evoked

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
picks = mne.pick_types(raw.info, meg='grad', exclude='bads')
reject = dict(grad=4000e-13)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, reject=reject, preload=True, proj=False)

epochs = epochs[:20]  # fewer samples to study regulrization

###############################################################################
# Compute covariance using automated regularization

# the best estimator in this list will be selected
method = ('empirical', 'shrunk', 'pca', 'factor_analysis')
noise_covs = compute_covariance(epochs, tmin=None, tmax=0, method=method,
                                return_estimators=True, projs=False,
                                verbose=True)

# the "return_estimator" flag returns all covariance estimators sorted by
# log-likelihood. Moreover the noise cov objects now contain extra info.

print('Covariance estimates sorted from best to worst')
for c in noise_covs:
    print("%s : %s" % (c['method'], c['loglik']))

###############################################################################
# Show whitening

# unwhitened evoked response

evoked = epochs.average()
evoked.plot()

picks = mne.pick_types(evoked.info, meg='grad', eeg=False, exclude='bads')

evokeds_white = [whiten_evoked(evoked, n, picks) for n in noise_covs]

evoked_white_best = evokeds_white[0]
evoked_white_worst = evokeds_white[-1]

# plot the whitened evoked data for to see if baseline signals match the
# assumption of Gaussian white noise from which we expect values around
# and less than 2 standard deviations. For the Global field power we expect
# a value of 1.

evoked_white_best.plot(unit=False, hline=[-2, 2])
evoked_white_worst.plot(unit=False, hline=[-2, 2])

# it's spatial whitening!
evoked_white_best.plot_topomap(ch_type='grad', scale=1, unit='Arb. U.',
                               contours=0, sensors=False)
evoked_white_worst.plot_topomap(ch_type='grad', scale=1, unit='Arb. U.',
                                contours=0, sensors=False)

fig_gfp, ax_gfp = plt.subplots(1)
times = evoked.times * 1e3

colors = [plt.cm.RdBu(i) for i in np.linspace(0.1, 0.8, 4)]

for evoked_white, noise_cov, color in zip(evokeds_white, noise_covs, colors):
    gfp = (evoked_white.data[picks] ** 2).sum(axis=0) / len(picks)
    ax_gfp.plot(times, gfp, label=noise_cov['method'], color=color)
    ax_gfp.set_xlabel('times [ms]')
    ax_gfp.set_ylabel('Global field power [chi^2]')
    ax_gfp.set_xlim(times[0], times[-1])
    ax_gfp.set_ylim(0, 20)

ax_gfp.axhline(1, color='red', linestyle='--',
               label='expected baseline (Gaussian)')
ax_gfp.legend(loc='upper right')
fig_gfp.show()
