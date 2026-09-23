"""Numba-accelerated helpers for :mod:`mne.stats.cluster_level`.

Kept in its own module so that importing the clustering code -- which happens via
``mne.label`` and therefore ``mne.coreg`` -- does not import numba.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._numba import jit


@jit()
def _masked_sum(x, c):
    return np.sum(x[c])


@jit()
def _masked_sum_power(x, c, t_power):
    return np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)


@jit()
def _sum_cluster_data(data, tstep):
    return np.sign(data) * tstep
