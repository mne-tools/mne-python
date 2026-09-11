"""Persistent decoded-data cache for Raw readers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

from .. import __version__ as MNE_VERSION  # ty: ignore[unresolved-import]
from ..utils import get_config, logger

_RAW_PRELOAD_CACHE_VERSION = 1


def _raw_preload_cache_info(raw):
    """Return the cache path and decoded array description."""
    cache_root = get_config("MNE_CACHE_DIR", None)
    if cache_root is None:
        raise ValueError(
            'preload="auto" requires a configured cache directory; use '
            "mne.set_cache_dir(...) first"
        )
    cache_dir = Path(cache_root).expanduser().resolve()
    cache_dir = cache_dir / f"raw-preload-v{_RAW_PRELOAD_CACHE_VERSION}"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    sources = []
    for filename in raw.filenames:
        if filename is None:
            raise ValueError(
                'preload="auto" requires stable source files; use preload=True '
                "or an explicit memory-map path"
            )
        path = Path(filename).resolve(strict=True)
        # some formats (e.g., CTF) name a directory rather than a single file
        members = sorted(path.rglob("*")) if path.is_dir() else [path]
        for member in members:
            if not member.is_file():
                continue
            result = member.stat()
            sources.append((str(member), int(result.st_size), int(result.st_mtime_ns)))

    dtype = np.dtype(raw._dtype)
    shape = (int(raw.info["nchan"]), int(raw.n_times))
    identity = (
        _RAW_PRELOAD_CACHE_VERSION,
        MNE_VERSION,
        type(raw).__module__,
        type(raw).__qualname__,
        sources,
        raw._raw_extras,
        raw._cals,
        dtype.str,
        shape,
    )
    try:
        key = hashlib.sha256(pickle.dumps(identity, protocol=5)).hexdigest()
    except Exception as exc:
        raise ValueError(
            f'preload="auto" cannot identify this {type(raw).__name__} source'
        ) from exc
    return cache_dir / f"{key}.data", sources, shape, dtype


def _raw_preload_cache_read(path, shape, dtype):
    """Map a complete decoded-data cache entry."""
    try:
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if path.stat().st_size != nbytes:
            return None
        return np.memmap(path, mode="c", dtype=dtype, shape=shape)
    except OSError:
        return None


def _raw_preload_auto(raw):
    """Reuse or create an automatic decoded-data cache entry."""
    path, sources, shape, dtype = _raw_preload_cache_info(raw)
    data = _raw_preload_cache_read(path, shape, dtype)
    if data is not None:
        logger.info(f"Reusing decoded data from {path}")
        return data

    # The temporary is per-process and os.replace is atomic, so concurrent
    # misses need no lock; they at worst decode the same entry twice.
    logger.info(f"Creating decoded data cache in {path.parent}")
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        data = np.memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        try:
            raw._read_segment(data_buffer=data)
            data.flush()
        finally:
            data._mmap.close()  # ty: ignore[unresolved-attribute]
        if _raw_preload_cache_info(raw)[1] != sources:
            raise RuntimeError(
                "Source data changed while decoded cache was created; retry"
            )
        try:
            os.replace(temporary, path)
        except OSError:
            # Windows refuses to replace an entry another process already mapped.
            pass
    finally:
        temporary.unlink(missing_ok=True)
    data = _raw_preload_cache_read(path, shape, dtype)
    if data is None:
        raise RuntimeError(f"Could not read back the decoded data cache at {path}")
    return data
