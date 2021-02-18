"""
==============================================================================
Compute iterative reweighted TF-MxNE with multiscale time-frequency dictionary
==============================================================================

The iterative reweighted TF-MxNE solver is a distributed inverse method
based on the TF-MxNE solver, which promotes focal (sparse) sources
:footcite:`StrohmeierEtAl2015`. The benefit of this approach is that:

  - it is spatio-temporal without assuming stationarity (sources properties
    can vary over time),
  - activations are localized in space, time and frequency in one step,
  - the solver uses non-convex penalties in the TF domain, which results in a
    solution less biased towards zero than when simple TF-MxNE is used,
  - using a multiscale dictionary allows to capture short transient
    activations along with slower brain waves :footcite:`BekhtiEtAl2016`.
"""
# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Yousra Bekhti <yousra.bekhti@gmail.com>
#         Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.datasets import somato
from mne.inverse_sparse import tf_mixed_norm, make_stc_from_dipoles
from mne.viz import plot_sparse_source_estimates

print(__doc__)


###############################################################################
# Load somatosensory MEG data

data_path = somato.data_path()
subject = '01'
task = 'somato'
raw_fname = op.join(data_path, 'sub-{}'.format(subject), 'meg',
                    'sub-{}_task-{}_meg.fif'.format(subject, task))
fwd_fname = op.join(data_path, 'derivatives', 'sub-{}'.format(subject),
                    'sub-{}_task-{}-fwd.fif'.format(subject, task))

condition = 'Unknown'

# Read evoked
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')
reject = dict(grad=4000e-13, eog=350e-6)
picks = mne.pick_types(raw.info, meg=True, eog=True)

event_id, tmin, tmax = 1, -1., 3.
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    reject=reject, preload=True)
evoked = epochs.filter(1, None).average()
evoked = evoked.pick_types(meg=True)
evoked.crop(tmin=0.008, tmax=0.2)

# Compute noise covariance matrix
cov = mne.compute_covariance(epochs, rank='info', tmax=0.)

# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

###############################################################################
# Run iterative reweighted multidict TF-MxNE solver

alpha, l1_ratio = 20, 0.05
loose, depth = 1, 0.95
# Use a multiscale time-frequency dictionary
wsize, tstep = [4, 16], [2, 4]


n_tfmxne_iter = 10
# Compute TF-MxNE inverse solution with dipole output
dipoles, residual = tf_mixed_norm(
    evoked, forward, cov, alpha=alpha, l1_ratio=l1_ratio,
    n_tfmxne_iter=n_tfmxne_iter, loose=loose,
    depth=depth, tol=1e-3,
    wsize=wsize, tstep=tstep, return_as_dipoles=True,
    return_residual=True)

# Crop to remove edges
for dip in dipoles:
    dip.crop(tmin=-0.05, tmax=0.3)
evoked.crop(tmin=-0.05, tmax=0.3)
residual.crop(tmin=-0.05, tmax=0.3)


###############################################################################
# Generate stc from dipoles
stc = make_stc_from_dipoles(dipoles, forward['src'])

plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="irTF-MxNE (cond %s)"
                             % condition)


###############################################################################
# Show the evoked response and the residual for gradiometers
ylim = dict(grad=[-300, 300])
evoked.pick_types(meg='grad')
evoked.plot(titles=dict(grad='Evoked Response: Gradiometers'), ylim=ylim,
            proj=True)

residual.pick_types(meg='grad')
residual.plot(titles=dict(grad='Residuals: Gradiometers'), ylim=ylim,
              proj=True)


###############################################################################
# References
# ----------
# .. footbibliography::
