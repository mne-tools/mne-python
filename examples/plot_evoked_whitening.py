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
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Reading
evoked = mne.fiff.Evoked(fname, setno=0, baseline=(None, 0), proj=True)
noise_cov = mne.read_cov(cov_fname)

###############################################################################
# Show result

  # Pick channels to view
picks = mne.fiff.pick_types(evoked.info, meg=True, eeg=True, exclude='bads')
evoked.plot(picks=picks)

noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                               grad=0.1, mag=0.1, eeg=0.1)

evoked_white = mne.whiten_evoked(evoked, noise_cov, picks, diag=True)

# plot the whitened evoked data to see if baseline signals match the
# assumption of Gaussian whiten noise from which we expect values around
# and less than 2 standard deviations.
import pylab as pl
pl.figure()
evoked_white.plot(picks=picks, unit=False, hline=[-2, 2])
