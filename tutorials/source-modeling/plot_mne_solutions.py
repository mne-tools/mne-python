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
from mne.minimum_norm import read_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

# Read data
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname_evoked, condition='Left Auditory',
                          baseline=(None, 0))
fname_inv = \
    data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif'

###############################################################################
# Fixed orientation
# -----------------
# First let's load a loose-orientation (``loose=0.2```) inverse, with the
# default depth weighting (0.8).

inv = read_inverse_operator(fname_inv)

###############################################################################
# Let's look at the current estimates using MNE. We'll take the absolute
# value of the source estimates to simplify the visualization.

snr = 3.0
lambda2 = 1.0 / snr ** 2
kwargs = dict(initial_time=0.08, hemi='both', subjects_dir=subjects_dir,
              size=(600, 600))

stc = apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True)
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)


###############################################################################
# Next let's use the default noise normalization, dSPM:

stc = apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True)
brain = stc.plot(figure=2, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)

###############################################################################
# And sLORETA:

stc = apply_inverse(evoked, inv, lambda2, 'sLORETA', verbose=True)
brain = stc.plot(figure=3, **kwargs)
brain.add_text(0.1, 0.9, 'sLORETA', 'title', font_size=14)

###############################################################################
# And finally eLORETA:

stc = apply_inverse(evoked, inv, lambda2, 'eLORETA', verbose=True)
brain = stc.plot(figure=4, **kwargs)
brain.add_text(0.1, 0.9, 'eLORETA', 'title', font_size=14)
