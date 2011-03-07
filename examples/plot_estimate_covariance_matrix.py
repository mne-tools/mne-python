"""
==============================================
Estimate covariance matrix from a raw FIF file
==============================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import os
import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = fiff.Raw(fname)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False

picks = fiff.pick_types(raw.info, meg=want_meg, eeg=want_eeg,
                        stim=want_stim, exclude=raw.info['bads'])

print "Number of picked channels : %d" % len(picks)

full_cov = mne.Covariance(kind='full')
full_cov.estimate_from_raw(raw, picks=picks)
print full_cov

diagonal_cov = mne.Covariance(kind='diagonal')
diagonal_cov.estimate_from_raw(raw, picks=picks)
print diagonal_cov

###############################################################################
# Show covariance
import pylab as pl
pl.figure(figsize=(8, 4))
pl.subplot(1, 2, 1)
pl.imshow(full_cov.data, interpolation="nearest")
pl.title('Full covariance matrix')
pl.subplot(1, 2, 2)
pl.imshow(diagonal_cov.data, interpolation="nearest")
pl.title('Diagonal covariance matrix')
pl.subplots_adjust(0.06, 0.02, 0.98, 0.94, 0.16, 0.26)
pl.show()
