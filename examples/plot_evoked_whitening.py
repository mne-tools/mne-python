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
picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
reject = dict(grad=4000e-13, mag=4e-12, eeg=80e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, reject=reject, preload=True, proj=False)

epochs = epochs[:20]  # fewer samples to study regulrization
# For your data, use as many samples as you can!

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

picks = mne.pick_types(evoked.info, meg=True, eeg=True, exclude='bads')

evokeds_white = [whiten_evoked(evoked, n, picks) for n in noise_covs]

# plot the whitened evoked data for to see if baseline signals match the
# assumption of Gaussian white noise from which we expect values around
# and less than 2 standard deviations. For the Global field power we expect
# a value of 1.

for evoked_white, quality in zip(evokeds_white[::3], ('best', 'worst')):
    fig = evoked_white.plot(unit=False, hline=[-2, 2])
    fig.suptitle('whitened evoked data (%s)' % quality)
    fig.subplots_adjust(top=0.9)
    fig.canvas.draw()

# it's spatial whitening! Can you see the sparkles for the worst?
for evoked_white, quality in zip(evokeds_white[::3], ('best', 'worst')):
    fig = evoked_white.plot_topomap(scale=1, unit='Arb. U.', contours=0,
                                    sensors=False)
    fig.suptitle('whitened topography (Magnetometers, %s)' % quality)
    fig.subplots_adjust(top=0.65, right=.95, bottom=0.12)
    fig.canvas.draw()

times = evoked.times * 1e3

fig_gfp, ax_gfp = plt.subplots(3, 1, sharex=True, sharey=True)

colors = [plt.cm.RdBu(i) for i in np.linspace(0.2, 0.8, 4)]


def whitened_gfp(x):
    """Whitened Global Field Power

    The MNE inverse solver assumes zero mean whitend data as input.
    Therefore, a chi^2 statistic will be best to detect model violations.
    """
    return np.sum(x ** 2, axis=0) / len(x)

fig_gfp.suptitle('Whitened global field power (GFP)')
for evoked_white, noise_cov, color in zip(evokeds_white, noise_covs, colors):
    i = 0
    for sub_picks in (mne.pick_types(evoked.info, meg='mag', eeg=False),
                      mne.pick_types(evoked.info, meg='grad', eeg=False),
                      mne.pick_types(evoked.info, meg=False, eeg=True)):

        gfp = whitened_gfp(evoked_white.data[sub_picks])
        ax_gfp[i].plot(times, gfp, label=noise_cov['method'], color=color)
        ax_gfp[i].set_xlabel('times [ms]')
        ax_gfp[i].set_ylabel('GFP [chi^2]')
        ax_gfp[i].set_xlim(times[0], times[-1])
        ax_gfp[i].set_ylim(0, 10)
        ax_gfp[i].axhline(1, color='red', linestyle='--')
        i += 1

ax_gfp[-1].legend(loc='upper right')
fig_gfp.show()
