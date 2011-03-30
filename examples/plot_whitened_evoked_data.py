"""
==================================================
Whiten evoked data using a noise covariance matrix
==================================================

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Reading
evoked = fiff.read_evoked(fname, setno=0, baseline=(None, 0))
cov = mne.Covariance()
cov.load(cov_fname)

evoked_whiten, W = cov.whiten_evoked(evoked)

bads = evoked_whiten.info['bads']
ind_meg_grad = fiff.pick_types(evoked.info, meg='grad', exclude=bads)
ind_meg_mag = fiff.pick_types(evoked.info, meg='mag', exclude=bads)
ind_eeg = fiff.pick_types(evoked.info, meg=False, eeg=True, exclude=bads)

###############################################################################
# Show result
import pylab as pl
pl.clf()
pl.subplot(3, 1, 1)
pl.plot(evoked.times, evoked_whiten.data[ind_meg_grad,:].T)
pl.title('MEG Planar Gradiometers')
pl.xlabel('time (s)')
pl.ylabel('MEG data')
pl.subplot(3, 1, 2)
pl.plot(evoked.times, evoked_whiten.data[ind_meg_mag,:].T)
pl.title('MEG Magnetometers')
pl.xlabel('time (s)')
pl.ylabel('MEG data')
pl.subplot(3, 1, 3)
pl.plot(evoked.times, evoked_whiten.data[ind_eeg,:].T)
pl.title('EEG')
pl.xlabel('time (s)')
pl.ylabel('EEG data')
pl.subplots_adjust(0.1, 0.08, 0.94, 0.94, 0.2, 0.63)
pl.show()
