"""Numba-accelerated BDF (24-bit little-endian) sample decoding.

Optional acceleration: falls back to the vectorized-numpy path in
``mne.io.edf.edf._read_ch`` when numba is unavailable.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._numba import jit


@jit()
def decode_int24(buf):  # pragma: no cover
    """Decode packed 24-bit little-endian samples to int32.

    ``buf`` is a ``(n_samples, 3)`` uint8 array whose rows hold the low,
    middle, and high bytes of each signed sample.
    """
    n = buf.shape[0]
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        # plain-Python integer arithmetic so the non-numba fallback follows
        # the same semantics as the jitted version (values stay within
        # [-2**23, 2**23) after the sign fix, so int32 stores never overflow)
        v = int(buf[i, 0]) | (int(buf[i, 1]) << 8) | (int(buf[i, 2]) << 16)
        if v >= (1 << 23):
            v -= 1 << 24
        out[i] = v
    return out
