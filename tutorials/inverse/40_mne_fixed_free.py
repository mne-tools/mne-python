# -*- coding: utf-8 -*-
"""
.. _tut-mne-fixed-free:

===============================
Computing various MNE solutions
===============================

This example shows example fixed- and free-orientation source localizations
produced by MNE, dSPM, sLORETA, and eLORETA.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

# Read data (just MEG here for speed, though we could use MEG+EEG)
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                          baseline=(None, 0))
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fwd = mne.read_forward_solution(fname_fwd)
cov = mne.read_cov(fname_cov)
# crop for speed in these examples
evoked.crop(0.05, 0.15)

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
kwargs = dict(initial_time=0.08, hemi='both', subjects_dir=subjects_dir,
              size=(600, 600))

stc = abs(apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

###############################################################################
# Next let's use the default noise normalization, dSPM:

stc = abs(apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True))
brain = stc.plot(figure=2, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)

###############################################################################
# And sLORETA:

stc = abs(apply_inverse(evoked, inv, lambda2, 'sLORETA', verbose=True))
brain = stc.plot(figure=3, **kwargs)
brain.add_text(0.1, 0.9, 'sLORETA', 'title', font_size=14)

###############################################################################
# And finally eLORETA:

stc = abs(apply_inverse(evoked, inv, lambda2, 'eLORETA', verbose=True))
brain = stc.plot(figure=4, **kwargs)
brain.add_text(0.1, 0.9, 'eLORETA', 'title', font_size=14)
del inv

###############################################################################
# Free orientation
# ----------------
# Now let's not constrain the orientation of the dipoles at all by creating
# a free-orientation inverse.

inv = make_inverse_operator(evoked.info, fwd, cov, loose=1., depth=0.8,
                            verbose=True)
del fwd

###############################################################################
# Let's look at the current estimates using MNE. We'll take the absolute
# value of the source estimates to simplify the visualization.

stc = apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True)
brain = stc.plot(figure=5, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

###############################################################################
# Next let's use the default noise normalization, dSPM:

stc = apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True)
brain = stc.plot(figure=6, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)

###############################################################################
# sLORETA:

stc = apply_inverse(evoked, inv, lambda2, 'sLORETA', verbose=True)
brain = stc.plot(figure=7, **kwargs)
brain.add_text(0.1, 0.9, 'sLORETA', 'title', font_size=14)

###############################################################################
# And finally eLORETA:

stc = apply_inverse(evoked, inv, lambda2, 'eLORETA', verbose=True,
                    method_params=dict(eps=1e-4))  # larger eps just for speed
brain = stc.plot(figure=8, **kwargs)
brain.add_text(0.1, 0.9, 'eLORETA', 'title', font_size=14)
