"""
==================================
Sensitivity map of SSP projections
==================================

This example shows the sources that have a forward field
similar to the first SSP vector correcting for ECG.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg_proj.fif'

fwd = mne.read_forward_solution(fname, surf_ori=True)
projs = mne.read_proj(ecg_fname)
projs = projs[3:][::2]  # take only one projection per channel type

# Compute sensitivity map
ssp_ecg_map = mne.sensitivity_map(fwd, ch_type='grad', projs=projs,
                                  mode='angle')

###############################################################################
# Show sensitivy map

import pylab as pl
pl.hist(ssp_ecg_map.data.ravel())
pl.show()

args = dict(fmin=0.2, fmid=0.6, fmax=1., smoothing_steps=7, hemi='rh')
ssp_ecg_map.plot(subject='sample', time_label='ECG SSP sensitivity', **args)
