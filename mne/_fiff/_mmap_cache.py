"""PID-keyed memmap caching for direct byte-offset reads.

Used by readers that need random access into raw data files (currently the
FIF raw reader). Keyed by PID so forked worker processes (e.g., PyTorch
DataLoader workers) create their own mapping instead of sharing a parent's,
and validated against file size/mtime so stale mappings are never reused.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import numpy as np

_MAX_CACHE = 16
_cache = {}


def get_u8_memmap(path):
    """Return a uint8 memmap of *path* (PID-keyed), or None on any failure."""
    try:
        st = os.stat(path)
        key = (os.getpid(), str(path))
        hit = _cache.get(key)
        if hit is not None:
            mm, mtime_ns, size = hit
            if mtime_ns == st.st_mtime_ns and size == st.st_size:
                return mm
            _cache.pop(key, None)
        mm = np.memmap(str(path), dtype=np.uint8, mode="r")
    except Exception:
        return None
    _cache[key] = (mm, st.st_mtime_ns, st.st_size)
    while len(_cache) > _MAX_CACHE:
        _cache.pop(next(iter(_cache)))
    return mm
