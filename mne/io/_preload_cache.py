"""Persistent decoded-data cache for Raw readers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import hashlib
import json
import os
import pickle
import stat
from pathlib import Path

import numpy as np

from .. import __version__ as MNE_VERSION  # ty: ignore[unresolved-import]
from ..utils import _soft_import, get_config, logger

_RAW_PRELOAD_CACHE_VERSION = 1
_RAW_PRELOAD_LOCK_TIMEOUT = 300.0


def _raw_preload_open_regular(path):
    """Open and validate a regular cache file."""
    file = open(path, "rb")
    try:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise OSError("Decoded data cache entries must be regular files")
    except Exception:
        file.close()
        raise
    return file


def _raw_preload_source_signature(raw):
    """Return filesystem identities for the source data files."""
    sources = []
    for filename in raw.filenames:
        if filename is None:
            raise ValueError(
                'preload="auto" requires stable source files; use preload=True '
                "or an explicit memory-map path"
            )
        path = Path(filename).resolve(strict=True)
        if path.suffix == ".gz":
            raise ValueError(
                'preload="auto" supports only uncompressed source files; use '
                "preload=True for compressed files"
            )
        result = path.stat()
        if not stat.S_ISREG(result.st_mode):
            raise OSError("Raw source data must be regular files")
        # ponytail: hash contents only if path, size, and mtime prove insufficient.
        sources.append((str(path), int(result.st_size), int(result.st_mtime_ns)))
    return sources


def _raw_preload_cache_dir(cache_root=None):
    """Resolve and validate the managed cache directory."""
    if cache_root is None:
        cache_root = get_config("MNE_CACHE_DIR", None)
    if cache_root is None:
        raise ValueError(
            'preload="auto" requires a configured cache directory; use '
            "mne.set_cache_dir(...) first"
        )
    cache_root = Path(cache_root).expanduser().resolve()
    cache_dir = cache_root / f"raw-preload-v{_RAW_PRELOAD_CACHE_VERSION}"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise OSError(f"Decoded data cache must be a regular directory: {cache_dir}")
    return cache_dir


def _raw_preload_cache_info(raw):
    """Return the managed cache location and expected array description."""
    cache_dir = _raw_preload_cache_dir()
    sources = _raw_preload_source_signature(raw)
    dtype = np.dtype(raw._dtype)
    shape = (int(raw.info["nchan"]), int(raw.n_times))
    identity = dict(
        version=_RAW_PRELOAD_CACHE_VERSION,
        mne_version=MNE_VERSION,
        reader=(type(raw).__module__, type(raw).__qualname__),
        sources=sources,
        raw_extras=raw._raw_extras,
        read_picks=raw._read_picks,
        cals=raw._cals,
        projector=raw._projector,
        compensator=raw._comp,
        first_samps=raw._first_samps,
        last_samps=raw._last_samps,
        dtype=dtype.str,
        shape=shape,
    )
    try:
        serialized = pickle.dumps(identity, protocol=5)
    except Exception as exc:
        raise ValueError(
            f'preload="auto" cannot identify this {type(raw).__name__} source'
        ) from exc
    key = hashlib.sha256(serialized).hexdigest()
    return cache_dir, key, sources, shape, dtype


def _raw_preload_generation_valid(name, key):
    """Check that a manifest generation is a managed basename."""
    prefix = f"{key}."
    suffix = ".data"
    if (
        not isinstance(name, str)
        or not name.startswith(prefix)
        or not name.endswith(suffix)
    ):
        return False
    token = name[len(prefix) : -len(suffix)]
    return len(token) == 32 and all(char in "0123456789abcdef" for char in token)


def _raw_preload_read_manifest(cache_dir, key):
    """Read one manifest through its validated handle."""
    path = cache_dir / f"{key}.json"
    with _raw_preload_open_regular(path) as file:
        if os.fstat(file.fileno()).st_size > 4096:
            raise ValueError("Oversized Raw preload manifest")
        return json.loads(file.read().decode("utf-8"))


def _raw_preload_cache_read(raw, cache_dir, key, sources, shape, dtype):
    """Read and validate one managed decoded-data cache entry."""
    try:
        manifest = _raw_preload_read_manifest(cache_dir, key)
        if set(manifest) != {"generation"}:
            return None
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if not _raw_preload_generation_valid(manifest["generation"], key):
            return None
        generation = cache_dir / manifest["generation"]
        with _raw_preload_open_regular(generation) as file:
            if os.fstat(file.fileno()).st_size != nbytes:
                return None
            data = np.memmap(file, mode="c", dtype=dtype, shape=shape)
            data.filename = str(generation)  # ty: ignore[invalid-assignment]
        if _raw_preload_source_signature(raw) != sources:
            data._mmap.close()  # ty: ignore[unresolved-attribute]  # memmap private
            return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    logger.info(f"Reusing decoded data from {generation}")
    return data


def _raw_preload_scavenge_key(cache_dir, key):
    """Remove abandoned temporary and unreferenced same-key generations."""
    referenced = None
    try:
        manifest = _raw_preload_read_manifest(cache_dir, key)
        candidate = manifest.get("generation")
        if _raw_preload_generation_valid(candidate, key):
            referenced = candidate
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    patterns = (f".{key}.*.tmp", f"{key}.*.data")
    for pattern in patterns:
        for path in cache_dir.glob(pattern):
            if path.name == referenced:
                continue
            try:
                path.unlink()
            except OSError:
                logger.debug(
                    f"Could not remove abandoned Raw preload cache file {path}"
                )


def _raw_preload_cache_create(raw, cache_dir, key, sources, shape, dtype):
    """Decode, durably publish, and reopen an immutable cache generation."""
    token = os.urandom(16).hex()
    generation_name = f"{key}.{token}.data"
    generation = cache_dir / generation_name
    temporary = cache_dir / f".{generation_name}.tmp"
    manifest_temporary = None
    manifest_published = False
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+b") as file:
            file.truncate(nbytes)
            data = np.memmap(file, mode="r+", dtype=dtype, shape=shape)
            try:
                raw._read_segment(data_buffer=data)
                data.flush()
            finally:
                data._mmap.close()  # ty: ignore[unresolved-attribute]  # memmap private
            os.fsync(file.fileno())
        if _raw_preload_source_signature(raw) != sources:
            raise RuntimeError(
                "Source data changed while decoded cache was created; retry"
            )
        os.replace(temporary, generation)
        manifest = dict(generation=generation_name)
        manifest_temporary = cache_dir / f".{key}.{os.urandom(16).hex()}.json.tmp"
        descriptor = os.open(
            manifest_temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(manifest_temporary, cache_dir / f"{key}.json")
        manifest_published = True
        with _raw_preload_open_regular(generation) as file:
            result = np.memmap(file, mode="c", dtype=dtype, shape=shape)
            result.filename = str(generation)  # ty: ignore[invalid-assignment]
        return result
    finally:
        for path in (temporary, manifest_temporary):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        if not manifest_published:
            try:
                generation.unlink(missing_ok=True)
            except OSError:
                pass


def _raw_preload_auto(raw):
    """Reuse or create an automatic decoded-data cache entry."""
    cache_dir, key, sources, shape, dtype = _raw_preload_cache_info(raw)
    key_lock = cache_dir / f"{key}.lock"
    data = _raw_preload_cache_read(raw, cache_dir, key, sources, shape, dtype)
    if data is not None:
        return data
    # Importing filelock is measurable, so keep it off the cache-hit path.
    filelock = _soft_import("filelock", "locking the decoded-data cache")

    with filelock.FileLock(key_lock, timeout=_RAW_PRELOAD_LOCK_TIMEOUT):
        _raw_preload_scavenge_key(cache_dir, key)
        data = _raw_preload_cache_read(raw, cache_dir, key, sources, shape, dtype)
        if data is None:
            logger.info(f"Creating decoded data cache in {cache_dir}")
            data = _raw_preload_cache_create(raw, cache_dir, key, sources, shape, dtype)
        _raw_preload_scavenge_key(cache_dir, key)
    return data
