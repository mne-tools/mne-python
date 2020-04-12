# -*- coding: utf-8 -*-
"""
======================================
Show noise levels from empty room data
======================================

This shows how to use :meth:`mne.io.Raw.plot_psd` to examine noise levels
of systems. See [1]_ for an example.

References
----------
.. [1] Khan S, Cohen D (2013). Note: Magnetic noise from the inner wall of
   a magnetically shielded room. Review of Scientific Instruments 84:56101.
   https://doi.org/10.1063/1.4802845
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import mne

data_path = mne.datasets.sample.data_path()

raw_erm = mne.io.read_raw_fif(op.join(data_path, 'MEG', 'sample',
                                      'ernoise_raw.fif'), preload=True)

###############################################################################
# We can plot the absolute noise levels:
raw_erm.plot_psd(tmax=10., average=True, spatial_colors=False,
                 dB=False, xscale='log')
