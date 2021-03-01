"""

.. _ex-covariance-whitening-dspm:

===================================================
Demonstrate impact of whitening on source estimates
===================================================

This example demonstrates the relationship between the noise covariance
estimate and the MNE / dSPM source amplitudes. It computes source estimates for
the SPM faces data and compares proper regularization with insufficient
regularization based on the methods described in
:footcite:`EngemannGramfort2015`. This example demonstrates
that improper regularization can lead to overestimation of source amplitudes.
This example makes use of the previous, non-optimized code path that was used
before implementing the suggestions presented in
:footcite:`EngemannGramfort2015`.

This example does quite a bit of processing, so even on a
fast machine it can take a couple of minutes to complete.

.. warning:: Please do not copy the patterns presented here for your own
             analysis, this is example is purely illustrative.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import spm_face
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.cov import compute_covariance

print(__doc__)

##############################################################################
# Get data

data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'

raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D.ds'

raw = io.read_raw_ctf(raw_fname % 1)  # Take first run
# To save time and memory for this demo, we'll just use the first
# 2.5 minutes (all we need to get 30 total events) and heavily
# resample 480->60 Hz (usually you wouldn't do either of these!)
raw.crop(0, 150.).pick_types(meg=True, stim=True, exclude='bads').load_data()
raw.filter(None, 20.)

events = mne.find_events(raw, stim_channel='UPPT001')

event_ids = {"faces": 1, "scrambled": 2}
tmin, tmax = -0.2, 0.5
baseline = (None, 0)
reject = dict(mag=3e-12)

# inverse parameters
conditions = 'faces', 'scrambled'
snr = 3.0
lambda2 = 1.0 / snr ** 2
clim = dict(kind='value', lims=[0, 2.5, 5])

###############################################################################
# Estimate covariances

samples_epochs = 5, 15,
method = 'empirical', 'shrunk'
colors = 'steelblue', 'red'
epochs = mne.Epochs(
    raw, events, event_ids, tmin, tmax,
    baseline=baseline, preload=True, reject=reject, decim=8)
del raw

noise_covs = list()
evokeds = list()
stcs = list()
methods_ordered = list()
for n_train in samples_epochs:
    # estimate covs based on a subset of samples
    # make sure we have the same number of conditions.
    idx = np.sort(np.concatenate([
        np.where(epochs.events[:, 2] == event_ids[cond])[0][:n_train]
        for cond in conditions]))
    epochs_train = epochs[idx]
    epochs_train.equalize_event_counts(event_ids)
    assert len(epochs_train) == 2 * n_train

    # We know some of these have too few samples, so suppress warning
    # with verbose='error'
    noise_covs.append(compute_covariance(
        epochs_train, method=method, tmin=None, tmax=0,  # baseline only
        return_estimators=True, rank=None, verbose='error'))  # returns list
    # prepare contrast
    evokeds.append([epochs_train[k].average() for k in conditions])
    del epochs_train
del epochs

# Make forward
trans = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
# oct5 and add_dist are just for speed, not recommended in general!
src = mne.setup_source_space(
    'spm', spacing='oct5', subjects_dir=data_path + '/subjects',
    add_dist=False)
bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
forward = mne.make_forward_solution(evokeds[0][0].info, trans, src, bem)
del src
for noise_covs_, evokeds_ in zip(noise_covs, evokeds):
    # do contrast

    # We skip empirical rank estimation that we introduced in response to
    # the findings in reference [1] to use the naive code path that
    # triggered the behavior described in [1]. The expected true rank is
    # 274 for this dataset. Please do not do this with your data but
    # rely on the default rank estimator that helps regularizing the
    # covariance.
    stcs.append(list())
    methods_ordered.append(list())
    for cov in noise_covs_:
        inverse_operator = make_inverse_operator(
            evokeds_[0].info, forward, cov, loose=0.2, depth=0.8)
        assert len(inverse_operator['sing']) == 274  # sanity check
        stc_a, stc_b = (apply_inverse(e, inverse_operator, lambda2, "dSPM",
                                      pick_ori=None) for e in evokeds_)
        stc = stc_a - stc_b
        methods_ordered[-1].append(cov['method'])
        stcs[-1].append(stc)
    del inverse_operator, cov, stc, stc_a, stc_b
del forward, noise_covs, evokeds  # save some memory


##############################################################################
# Show the resulting source estimates

fig, (axes1, axes2) = plt.subplots(2, 3, figsize=(9.5, 5))

for ni, (n_train, axes) in enumerate(zip(samples_epochs, (axes1, axes2))):
    # compute stc based on worst and best
    ax_dynamics = axes[1]
    for stc, ax, method, kind, color in zip(stcs[ni],
                                            axes[::2],
                                            methods_ordered[ni],
                                            ['best', 'worst'],
                                            colors):
        brain = stc.plot(subjects_dir=subjects_dir, hemi='both', clim=clim,
                         initial_time=0.175, background='w', foreground='k')
        brain.show_view('ven')
        im = brain.screenshot()
        brain.close()

        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(im)
        ax.set_title('{0} ({1} epochs)'.format(kind, n_train * 2))

        # plot spatial mean
        stc_mean = stc.data.mean(0)
        ax_dynamics.plot(stc.times * 1e3, stc_mean,
                         label='{0} ({1})'.format(method, kind),
                         color=color)
        # plot spatial std
        stc_var = stc.data.std(0)
        ax_dynamics.fill_between(stc.times * 1e3, stc_mean - stc_var,
                                 stc_mean + stc_var, alpha=0.2, color=color)

    # signal dynamics worst and best
    ax_dynamics.set(title='{0} epochs'.format(n_train * 2),
                    xlabel='Time (ms)', ylabel='Source Activation (dSPM)',
                    xlim=(tmin * 1e3, tmax * 1e3), ylim=(-3, 3))
    ax_dynamics.legend(loc='upper left', fontsize=10)

fig.subplots_adjust(hspace=0.2, left=0.01, right=0.99, wspace=0.03)

###############################################################################
# References
# ----------
# .. footbibliography::
