"""Numba-accelerated helpers for :mod:`mne.utils.numerics`.

Kept in its own module so that importing ``mne.utils.numerics`` -- which happens
during ``import mne`` -- does not import numba.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._numba import has_numba, jit

if has_numba:

    @jit(fastmath=False)
    def _arange_div(n, d):  # noqa: D103
        out = np.empty(n, np.float64)
        for i in range(n):
            out[i] = i / d
        return out

else:  # pragma: no cover
    from .numerics import _arange_div_fallback

    _arange_div = _arange_div_fallback
