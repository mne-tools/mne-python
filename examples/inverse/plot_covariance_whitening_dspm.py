# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

"""Demonstrate impact of whitening on soucre estimates

This example computes dSPM source estimates for the SPM faces data and compares
proper with insuficcient regularization based on the methods described in [1].


References
----------
    [1] Engemann D. and Gramfort A. Automated model selection in covariance
    estimation and spatial whitening of MEG and EEG signals. (in press.)
    NeuroImage.
"""
print(__doc__)

import os
import os.path as op

import numpy as np
from scipy.misc import imread

import mne
from mne import io
from mne.datasets import spm_face
from mne.minimum_norm import apply_inverse, make_inverse_operator
import matplotlib.pyplot as plt
from mne.cov import compute_covariance

###############################################################################
# Get data

data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'

raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D_raw.fif'

raw = io.Raw(raw_fname % 1, preload=True)  # Take first run

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 30, method='iir', n_jobs=6)

events = mne.find_events(raw, stim_channel='UPPT001')

event_ids = {"faces": 1, "scrambled": 2}
tmin, tmax = -0.2, 0.5
baseline = None  # no baseline as high-pass is applied
reject = dict(mag=3e-12)

# Make source space

mri = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
src = mne.setup_source_space('spm', spacing='oct6', subjects_dir=subjects_dir,
                             overwrite=True, add_dist=False)
bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
forward = mne.make_forward_solution(raw.info, mri=mri, src=src, bem=bem)
forward = mne.convert_forward_solution(forward, surf_ori=True)

# inverse parameters
conditions = 'faces', 'scrambled'
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'
fmin, fmid, fmax, transp = 0, 2.5, 5, True

################################################################################
# Estimate covariance and show resulting source estimates
method = ['ec', 'sc', 'reg']
best_colors = 'red', 'steelblue',
samples_epochs = 5, 15,
fig, (axes1, axes2) = plt.subplots(2, 3, figsize=(12, 6.5))


def brain_to_mpl(brain):
    """convert image to be usable with matplotlib"""
    tmp_path = op.abspath(op.join(op.curdir, 'my_tmp'))
    brain.save_imageset(tmp_path, views=['ven'])
    im = imread(tmp_path + '_ven.png')
    os.remove(tmp_path + '_ven.png')
    return im

for n_train, (ax_stc_worst, ax_dynamics, ax_stc_best) in zip(samples_epochs,
                                                             (axes1, axes2,)):
    # estimate covs based on a subset of samples
    # make sure we have the same numner of conditions.
    events_ = np.concatenate([events[events[:, 2] == id_][:n_train]
                              for id_ in [event_ids[k] for k in conditions]])
    epochs_train = mne.Epochs(raw, events_, event_ids, tmin, tmax, picks=picks,
                              baseline=baseline, preload=True, reject=reject)
    epochs_train.equalize_event_counts(event_ids, copy=False)

    noise_covs = compute_covariance(epochs_train, method=method,
                                    tmin=None, tmax=0,  # baseline only
                                    return_estimators=True)  # returns dict here
    # determine worst and best
    ests_keys, scores = zip(*[(k, v['loglik']) for k, v in noise_covs.items()])
    best, worst = [ests_keys[f(scores)] for f in [np.argmax, np.argmin]]

    # prepare contrast
    evokeds = [epochs_train[k].average() for k in conditions]

    # compute stc based on worst and best
    for est, ax, kind, color in zip([worst, best], (ax_stc_worst, ax_stc_best),
                                    ['worst', 'best'], best_colors):
        inverse_operator = make_inverse_operator(epochs_train.info, forward,
                                                 noise_covs[est]['cov'],
                                                 loose=0.2, depth=0.8)
        stc_a, stc_b = (apply_inverse(e, inverse_operator, lambda2, "dSPM",
                                      pick_ori=None) for e in evokeds)
        stc = stc_a - stc_b
        brain = stc.plot(subjects_dir=subjects_dir, hemi='both',
                         colormap='hot')
        brain.scale_data_colormap(fmin, fmid, fmax, transp)
        brain.set_time(175)

        im = brain_to_mpl(brain)
        brain.close()
        ax.imshow(im)
        ax.set_title('{0} ({1} epochs)'.format(kind, n_train))
        ax.set_axis_off()

        # plot spatial mean
        stc_mean = stc.data.mean(0)
        ax_dynamics.plot(stc.times * 1e3, stc_mean,
                         label='{0} ({1})'.format(est, kind),
                         color=color)
        # plot spatial std
        stc_var = stc.data.std(0)
        ax_dynamics.fill_between(stc.times * 1e3, stc_mean - stc_var,
                                 stc_mean + stc_var, alpha=0.2, color=color)

    # signal dynamics worst and best
    ax_dynamics.set_title('{0} epochs'.format(n_train * 2))
    ax_dynamics.set_xlabel('Time (ms)')
    ax_dynamics.set_ylabel('Source Activation (dSPM)')
    ax_dynamics.set_xlim(tmin * 1e3, tmax * 1e3)
    ax_dynamics.set_ylim(-3, 3)
    ax_dynamics.legend(loc='upper left')

fig.subplots_adjust(hspace=0.3)
fig.canvas.draw()
fig.show()
