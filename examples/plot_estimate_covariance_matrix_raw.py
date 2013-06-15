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

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = fiff.Raw(fname)

include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                                            include=include, exclude='bads')
# setup rejection
reject = dict(eeg=80e-6, eog=150e-6)

# Compute the covariance from the raw data
cov = mne.compute_raw_data_covariance(raw, picks=picks, reject=reject)
print cov

###############################################################################
# Show covariance
mne.viz.plot_cov(cov, raw.info, colorbar=True, proj=True)
# try setting proj to False to see the effect
