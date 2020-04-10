"""
==================================
Sensitivity map of SSP projections
==================================

This example shows the sources that have a forward field
similar to the first SSP vector correcting for ECG.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from mne import read_forward_solution, read_proj, sensitivity_map

from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

subjects_dir = data_path + '/subjects'
fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg-proj.fif'

fwd = read_forward_solution(fname)

projs = read_proj(ecg_fname)
# take only one projection per channel type
projs = projs[::2]

# Compute sensitivity map
ssp_ecg_map = sensitivity_map(fwd, ch_type='grad', projs=projs, mode='angle')

###############################################################################
# Show sensitivity map

plt.hist(ssp_ecg_map.data.ravel())
plt.show()

args = dict(clim=dict(kind='value', lims=(0.2, 0.6, 1.)), smoothing_steps=7,
            hemi='rh', subjects_dir=subjects_dir)
ssp_ecg_map.plot(subject='sample', time_label='ECG SSP sensitivity', **args)
