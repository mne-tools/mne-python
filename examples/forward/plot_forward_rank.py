"""
====================================================
Display the effective number of leadfield components
====================================================

Summarizing the diversity of spatial patterns contained
in the forward model can give us some idea about the spatial
resolution for a particular subject. Here we use
principal component analysis to explore the effective
dimensionality of the forward model.

To get started with forward modeling see :ref:`tut_forward`.

"""
# Author: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

subjects_dir = data_path + '/subjects'

# Read the forward solutions with surface orientation
fwd = mne.read_forward_solution(fwd_fname)

leadfield = fwd['sol']['data']
print("Leadfield size : %d x %d" % leadfield.shape)

###############################################################################
# Let's get the cumulative explained variance by sensor type


def _get_var_exp(data):
    """Do PCA and look at cumulative explained variance."""
    U, s, V = linalg.svd(data, full_matrices=False)
    s_ = s ** 2
    exp_var_ratio = s_.cumsum() / s_.cumsum().max()
    return exp_var_ratio


picks_mag = mne.pick_types(fwd['info'], meg='mag')
picks_grad = mne.pick_types(fwd['info'], meg='grad')
picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True)

# We will try to look at the same number of dimensions to
# avoid looking at trivial differences. We will use spatial
# subsampling for that purpose.
n_chans = 50
n_draws = 30
threshold = .99

plt.figure(figsize=(8, 6))
for picks, ch_type, color in zip(
        (picks_grad, picks_mag, picks_eeg),
        ('grad', 'mag', 'eeg'),
        ('c', 'b', 'k')):

    # use same number of channels while keeping spatial info.
    exp_var_ratio = np.array([
        _get_var_exp(
            leadfield[np.random.choice(picks, size=n_chans, replace=False)])
        for _ in range(n_draws)])

    exp_var_ratio_mean = np.mean(exp_var_ratio, axis=0)
    approx_rank = np.sum(exp_var_ratio_mean < threshold)
    plt.plot(exp_var_ratio_mean, color=color, linewidth=1,
             label=r'%s (rank$_{vexp%d}$ = %d)' % (
                 ch_type, threshold * 100, approx_rank))
    plt.plot(exp_var_ratio.T, color=color, linewidth=1,
             alpha=0.3)
plt.legend(loc='lower right')

# One can see that despite the same number of channels used,
# the gradiometer dimensionality saturates slower while eeg
# saturates fastest, leading to, respectively, more or less.
# independent patterns.
