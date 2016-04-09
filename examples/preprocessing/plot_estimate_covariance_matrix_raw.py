"""
==============================================
Estimate covariance matrix from a raw FIF file
==============================================

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = io.read_raw_fif(fname)

include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

# pick EEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
# setup rejection
reject = dict(eeg=80e-6, eog=150e-6)

# Compute the covariance from the raw data
cov = mne.compute_raw_covariance(raw, picks=picks, reject=reject)
print(cov)

###############################################################################
# Show covariance
fig_cov, fig_svd = mne.viz.plot_cov(cov, raw.info, colorbar=True, proj=True)
# try setting proj to False to see the effect
