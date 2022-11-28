# -*- coding: utf-8 -*-
"""
.. _ex-noise-level:

======================================
Show noise levels from empty room data
======================================

This shows how to use :meth:`mne.io.Raw.plot_psd` to examine noise levels
of systems. See :footcite:`KhanCohen2013` for an example.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import mne

data_path = mne.datasets.sample.data_path()

raw_erm = mne.io.read_raw_fif(
    data_path / 'MEG' / 'sample' / 'ernoise_raw.fif', preload=True
)

# %%
# We can plot the absolute noise levels:
raw_erm.plot_psd(tmax=10., average=True, spatial_colors=False,
                 dB=False, xscale='log')
# %%
# References
# ----------
#
# .. footbibliography::
