# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors

import os
from pathlib import Path

from ..._fiff.open import _NoCloseRead
from ...utils import _file_like, _validate_type, logger

# Persistent read handles for EDF/BDF/GDF files. Readers seek before every
# read, so a shared handle is safe; keying by PID keeps forked worker
# processes (e.g., PyTorch DataLoader workers) from sharing file-offset state
# through an inherited descriptor.
_HANDLE_CACHE = {}
_MAX_HANDLES = 8


class _NoCloseCached(_NoCloseRead):
    """A file object whose context manager detaches instead of closing.

    Used for handles shared through the per-process LRU cache: leaving the
    reader's ``with`` block must not close a descriptor other reads may still
    use.
    """

    def close(self):  # noqa: D102
        pass

    def __exit__(self, *args):  # noqa: D105
        # detach rather than close; the cache owns the lifetime
        return False


def _get_cached_fid(fname):
    """Return a persistent binary handle for *fname* (per process)."""
    key = (os.getpid(), str(fname))
    hit = _HANDLE_CACHE.get(key)
    if hit is not None:
        hit.seek(0)  # match fresh-open semantics
        return hit
    fid = open(fname, "rb")
    cached = _NoCloseCached(fid)
    _HANDLE_CACHE[key] = cached
    while len(_HANDLE_CACHE) > _MAX_HANDLES:
        old_key = next(iter(_HANDLE_CACHE))
        try:
            _HANDLE_CACHE.pop(old_key).fid.close()
        except Exception:
            pass
    return cached


def _gdf_edf_get_fid(fname, **kwargs):
    """Open a EDF/BDF/GDF file with no additional parsing."""
    if _file_like(fname):
        logger.debug("Using file-like I/O")
        fid = _NoCloseRead(fname)
        fid.seek(0)
        return fid
    _validate_type(fname, [Path, str], "fname", extra="or file-like")
    logger.debug("Using normal I/O")
    kwargs.pop("buffering", None)  # cached handle manages its own buffering
    return _get_cached_fid(Path(fname))
