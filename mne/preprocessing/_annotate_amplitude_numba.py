"""Numba-accelerated helper for :mod:`mne.preprocessing._annotate_amplitude`.

Kept in its own module so that importing ``mne.preprocessing`` does not import numba.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from .._numba import jit


@jit()
def _mark_inner(arr_k, onsets, offsets, min_duration_samples):
    """Inner loop of _reject_short_segments()."""
    for start, stop in zip(onsets, offsets):
        if stop - start < min_duration_samples:
            arr_k[start:stop] = False
