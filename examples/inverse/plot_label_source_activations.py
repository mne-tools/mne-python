"""
====================================================
Extracting the time series of activations in a label
====================================================

We first apply a dSPM inverse operator to get signed activations in a label
(with positive and negative values) and we then compare different strategies
to average the times series in a label. We compare a simple average, with an
averaging using the dipoles normal (flip mode) and then a PCA,
also using a sign flip.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
label = 'Aud-lh'
label_fname = data_path + '/MEG/sample/labels/%s.label' % label
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# Compute inverse solution
pick_ori = "normal"  # Get signed values to see the effect of sign filp
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori=pick_ori)

label = mne.read_label(label_fname)

stc_label = stc.in_label(label)
modes = ('mean', 'mean_flip', 'pca_flip')
tcs = dict()
for mode in modes:
    tcs[mode] = stc.extract_label_time_course(label, src, mode=mode)
print("Number of vertices : %d" % len(stc_label.data))

# View source activations
fig, ax = plt.subplots(1)
t = 1e3 * stc_label.times
ax.plot(t, stc_label.data.T, 'k', linewidth=0.5)
for mode, tc in tcs.items():
    ax.plot(t, tc[0], linewidth=3, label=str(mode))
ax.legend(loc='upper right')
ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
       title='Activations in Label : %s' % label)
