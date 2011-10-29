"""
==============================================
Estimate covariance matrix from a raw FIF file
==============================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = fiff.Raw(fname)

# Compute the covariance from the raw data
cov = mne.compute_raw_data_covariance(raw, reject=dict(eeg=80e-6, eog=150e-6))
print cov

bads = raw.info['bads']
sel_eeg = mne.fiff.pick_types(raw.info, meg=False, eeg=True, exclude=bads)
sel_mag = mne.fiff.pick_types(raw.info, meg='mag', eeg=False, exclude=bads)
sel_grad = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, exclude=bads)
idx_eeg = [cov.ch_names.index(raw.ch_names[c]) for c in sel_eeg]
idx_mag = [cov.ch_names.index(raw.ch_names[c]) for c in sel_mag]
idx_grad = [cov.ch_names.index(raw.ch_names[c]) for c in sel_grad]

###############################################################################
# Show covariance
import pylab as pl
pl.figure(figsize=(7.3, 2.7))
pl.subplot(1, 3, 1)
pl.imshow(cov.data[idx_eeg][:, idx_eeg], interpolation="nearest")
pl.title('EEG covariance')
pl.subplot(1, 3, 2)
pl.imshow(cov.data[idx_grad][:, idx_grad], interpolation="nearest")
pl.title('Gradiometers')
pl.subplot(1, 3, 3)
pl.imshow(cov.data[idx_mag][:, idx_mag], interpolation="nearest")
pl.title('Magnetometers')
pl.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
pl.show()
