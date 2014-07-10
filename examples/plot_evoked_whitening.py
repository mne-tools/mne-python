"""
=============================================
Whitening evoked data with a noise covariance
=============================================

Evoked data are loaded and then whitened using a given
noise covariance matrix. It's an excellent
quality check to see if baseline signals match the assumption
of Gaussian whiten noise from which we expect values around
and less than 2 standard deviations.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

from mne import read_cov, whiten_evoked, pick_types, read_evokeds
from mne.cov import regularize
from mne.datasets import sample

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Reading
evoked = read_evokeds(fname, condition=0, baseline=(None, 0), proj=True)
noise_cov = read_cov(cov_fname)

###############################################################################
# Show result

  # Pick channels to view
picks = pick_types(evoked.info, meg=True, eeg=True, exclude='bads')
evoked.plot(picks=picks)

noise_cov = regularize(noise_cov, evoked.info, grad=0.1, mag=0.1, eeg=0.1)

evoked_white = whiten_evoked(evoked, noise_cov, picks, diag=True)

# plot the whitened evoked data to see if baseline signals match the
# assumption of Gaussian whiten noise from which we expect values around
# and less than 2 standard deviations.
evoked_white.plot(picks=picks, unit=False, hline=[-2, 2])
