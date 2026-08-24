"""Numba-accelerated EDF/BDF digital-to-physical window decoding.

Optional acceleration: falls back to the vectorized-numpy path in
``mne.io.edf.edf._read_segment_file`` when numba is unavailable.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors

from ..._numba import jit


@jit(fastmath=False)
def decode_window(digital, cal_v, off_v, gain_v, out):  # pragma: no cover
    """Decode a block of digital samples to physical units.

    ``digital`` is a ``(k, n_blocks, buf_len)`` (possibly strided) integer
    view of raw digital samples; ``cal_v``, ``off_v``, and ``gain_v`` are
    length-``k`` float64 vectors; ``out`` is a ``(k, n_blocks * buf_len)``
    float64 array whose rows are filled with blocks concatenated along the
    sample axis such that::

        out[i, b * buf_len + j] = ((digital[i, b, j] * cal[i]) + off[i]) * gain[i]

    replicating exactly the operation order of the vectorized-numpy fallback
    (hence ``fastmath=False``: no FMA contraction or reassociation is
    allowed, so results are bit-identical to separate multiply/add rounding).
    """
    k = digital.shape[0]
    n_blk = digital.shape[1]
    n_smp = digital.shape[2]
    for i in range(k):
        cal = cal_v[i]
        off = off_v[i]
        gain = gain_v[i]
        out_i = out[i]
        for b in range(n_blk):
            base = b * n_smp
            for j in range(n_smp):
                out_i[base + j] = ((digital[i, b, j] * cal) + off) * gain
