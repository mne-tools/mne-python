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
from mne.viz import plot_evoked
from mne.datasets import sample

data_path = sample.data_path('.')
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Reading
evoked = fiff.read_evoked(fname, setno=0, baseline=(None, 0))
cov = mne.Covariance()
cov.load(cov_fname)

evoked_whiten, W = cov.whiten_evoked(evoked)

###############################################################################
# Show result
picks = fiff.pick_types(evoked_whiten.info, meg=True, eeg=True,
                    exclude=evoked_whiten.info['bads']) # Pick channels to view
plot_evoked(evoked_whiten, picks, unit=False) # plot without SI unit of data
