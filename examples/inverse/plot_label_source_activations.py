"""
====================================================
Extracting the time series of activations in a label
====================================================

We first apply a dSPM inverse operator to get signed activations
in a label (with positive and negative values) and we then
compare different strategies to average the times series
in a label. We compare a simple average, with an averaging
using the dipoles normal (flip mode) and then a PCA,
also using a sign flip.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.fiff import Evoked

data_path = sample.data_path()
label = 'Aud-lh'
label_fname = data_path + '/MEG/sample/labels/%s.label' % label
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# Compute inverse solution
pick_normal = True # Get signed values to see the effect of sign filp
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_normal=pick_normal)

label = mne.read_label(label_fname)

stc_label = stc.in_label(label)
mean = stc.extract_label_time_course(label, src, mode='mean')
mean_flip = stc.extract_label_time_course(label, src, mode='mean_flip')
pca = stc.extract_label_time_course(label, src, mode='pca_flip')

print "Number of vertices : %d" % len(stc_label.data)

# View source activations
import pylab as pl
pl.figure()
pl.plot(1e3 * stc_label.times, stc_label.data.T, 'k', linewidth=0.5)
h0, = pl.plot(1e3 * stc_label.times, mean.T, 'r', linewidth=3)
h1, = pl.plot(1e3 * stc_label.times, mean_flip.T, 'g', linewidth=3)
h2, = pl.plot(1e3 * stc_label.times, pca.T, 'b', linewidth=3)
pl.legend([h0, h1, h2], ['mean', 'mean flip', 'PCA flip'])
pl.xlabel('Time (ms)')
pl.ylabel('Source amplitude')
pl.title('Activations in Label : %s' % label)
pl.show()
