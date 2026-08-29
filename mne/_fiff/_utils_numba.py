"""Jitted helpers for :mod:`mne._fiff.utils`.

Kept in its own module so that importing mne does not import numba.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from .._numba import jit


@jit(fastmath=False)
def _scale_into(one, cals, data_view):
    """Cast, transpose and calibrate one raw buffer in a single pass.

    ``fastmath`` is off: this decodes stored values rather than approximating a
    computation, so reassociation buys nothing and only risks changing them.
    """
    n_channels, n_times = one.shape
    for ci in range(n_channels):
        cal = cals[ci]
        for ti in range(n_times):
            data_view[ci, ti] = one[ci, ti] * cal
