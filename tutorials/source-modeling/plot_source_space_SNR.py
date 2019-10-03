# -*- coding: utf-8 -*-
"""
.. _tut-mne-sourcespace-snr:

===============================
Computing source space SNR
===============================

This example shows how to compute and plot source space SNR.
"""
# Author: Padma Sundaram <tottochan@gmail.com>
#         Kaisu Lankinen <klankinen@mgh.harvard.edu>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

# Read data
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                          baseline=(None, 0))
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fwd = mne.read_forward_solution(fname_fwd)
cov = mne.read_cov(fname_cov)

###############################################################################
# Fixed orientation
# -----------------
# First let's create a fixed-orientation inverse, with the default weighting.

inv = make_inverse_operator(evoked.info, fwd, cov, loose=0., depth=0.8,
                            verbose=True)

###############################################################################
# Let's look at the current estimates using MNE. We'll take the absolute
# value of the source estimates to simplify the visualization.

snr = 3.0
lambda2 = 1.0 / snr ** 2

stc = apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True)
snr_stc = stc.estimate_snr(evoked.info, fwd, cov)

kwargs = dict(initial_time=0.29, hemi='both', subjects_dir=subjects_dir,
              size=(600, 600), colormap='viridis',
              clim=dict(kind='value', lims=(-20, 0, 20)))
snr_stc.plot(figure=1, **kwargs)
