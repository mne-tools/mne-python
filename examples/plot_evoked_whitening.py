"""
=============================================
Whitening evoked data with a noise covariance
=============================================

Evoked data are loaded and then whitened using a given
noise covariance matrix. It's an excellent
quality check to see if baseline signal match the assumption
of Gaussian whiten noise from which we expect values around
and less than 2 standard deviations.

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

from copy import deepcopy
import numpy as np
import mne
from mne.datasets import sample
from mne.viz import plot_evoked

data_path = sample.data_path('.')

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Reading
evoked = mne.fiff.Evoked(fname, setno=0, baseline=(None, 0), proj=True)
noise_cov = mne.read_cov(cov_fname)

###############################################################################
# Show result
picks = mne.fiff.pick_types(evoked.info, meg=True, eeg=True,
                        exclude=evoked.info['bads'])  # Pick channels to view

import pylab as pl
pl.close('all')
pl.figure()
plot_evoked(evoked, picks=picks)

from mne.cov import prepare_noise_cov

noise_cov = mne.cov.regularize(noise_cov, evoked.info, grad=0.1, mag=0.1, eeg=0.1)

def whiten(evoked, noise_cov, picks, diag=False):
    ch_names = [evoked.ch_names[k] for k in picks]
    n_chan = len(ch_names)
    evoked = deepcopy(evoked)

    if diag:
        noise_cov = deepcopy(noise_cov)
        noise_cov['data'] = np.diag(np.diag(noise_cov['data']))

    noise_cov = prepare_noise_cov(noise_cov, evoked.info, ch_names)

    W = np.zeros((n_chan, n_chan), dtype=np.float)
    #
    #   Omit the zeroes due to projection
    #
    eig = noise_cov['eig']
    nzero = (eig > 0)
    W[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
    #
    #   Rows of eigvec are the eigenvectors
    #
    W = np.dot(W, noise_cov['eigvec'])
    W = np.dot(noise_cov['eigvec'].T, W)
    evoked.data[picks] = np.sqrt(evoked.nave) * np.dot(W, evoked.data[picks])
    return evoked

evoked_white = whiten(evoked, noise_cov, picks, diag=True)
pl.figure()
plot_evoked(evoked_white, picks=picks, unit=False, hline=[-2, 2])
