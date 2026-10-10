"""Numba (optional requirement) helpers.

Kept out of :mod:`mne.fixes` so that ``import mne`` -- which pulls in
``mne.fixes`` via ``mne.utils.check`` -- does not pay the ~0.2 s cost of
``import numba``. Only modules that actually define jitted functions should
import from here, and they should be lazily imported submodules.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import numpy as np

# Here we choose different defaults to speed things up by default
try:
    import numba

    prange = numba.prange

    def jit(nopython=True, nogil=True, fastmath=True, cache=True, **kwargs):  # noqa
        return numba.jit(
            nopython=nopython, nogil=nogil, fastmath=fastmath, cache=cache, **kwargs
        )

except Exception:  # could be ImportError, SystemError, etc.
    has_numba = False
else:
    has_numba = os.getenv("MNE_USE_NUMBA", "true").lower() == "true"


if not has_numba:

    def jit(**kwargs):  # noqa
        def _jit(func):
            return func

        return _jit

    prange = range
    bincount = np.bincount

else:

    @jit()
    def bincount(x, weights, minlength):  # noqa: D103
        out = np.zeros(minlength)
        for idx, w in zip(x, weights):
            out[idx] += w
        return out
