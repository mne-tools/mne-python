"""Persistent decoded-data cache for Raw readers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ctypes
import errno
import hashlib
import json
import os
import pickle
import stat
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

from .. import __version__ as MNE_VERSION  # ty: ignore[unresolved-import]
from ..utils import get_config, logger

_RAW_PRELOAD_CACHE_VERSION = 1
_RAW_PRELOAD_LOCK_TIMEOUT = 300.0
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    import msvcrt

_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_ATTRIBUTE_TAG_INFO = 9
_FILE_BASIC_INFO = 0
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_ID_INFO = 18
_FILE_SHARE_ALL = 0x00000007
_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_PROCESS_SYNCHRONIZE = 0x00100000
_ERROR_INVALID_PARAMETER = 87
_WAIT_OBJECT_0 = 0


class _FileBasicInfo(ctypes.Structure):
    _fields_ = (
        ("creation_time", ctypes.c_longlong),
        ("last_access_time", ctypes.c_longlong),
        ("last_write_time", ctypes.c_longlong),
        ("change_time", ctypes.c_longlong),
        ("file_attributes", wintypes.DWORD),
    )


class _FileId128(ctypes.Structure):
    _fields_ = (("identifier", ctypes.c_ubyte * 16),)


class _FileIdInfo(ctypes.Structure):
    _fields_ = (
        ("volume_serial_number", ctypes.c_ulonglong),
        ("file_id", _FileId128),
    )


def _raw_preload_windows_fstat(descriptor, result):
    """Return Windows change time and file ID from an open handle."""
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    handle = getattr(msvcrt, "get_osfhandle")(descriptor)
    basic = _FileBasicInfo()
    get_ex = kernel32.GetFileInformationByHandleEx
    get_ex.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_ex.restype = wintypes.BOOL
    if not get_ex(handle, _FILE_BASIC_INFO, ctypes.byref(basic), ctypes.sizeof(basic)):
        raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
    file_info = _FileIdInfo()
    if not get_ex(
        handle, _FILE_ID_INFO, ctypes.byref(file_info), ctypes.sizeof(file_info)
    ):
        raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
    return dict(
        size=int(result.st_size),
        mtime_ns=int(basic.last_write_time) * 100,
        change_ns=int(basic.change_time) * 100,
        device=int(file_info.volume_serial_number),
        inode=int.from_bytes(bytes(file_info.file_id.identifier), "little"),
    )


def _raw_preload_fstat(descriptor):
    """Return an identity token for an already-open regular file."""
    result = os.fstat(descriptor)
    if not stat.S_ISREG(result.st_mode):
        raise OSError("Decoded data cache entries must be regular files")
    if _IS_WINDOWS:
        return _raw_preload_windows_fstat(descriptor, result)
    return dict(
        size=int(result.st_size),
        mtime_ns=int(result.st_mtime_ns),
        change_ns=int(result.st_ctime_ns),
        device=int(result.st_dev),
        inode=int(result.st_ino),
    )


def _raw_preload_open_windows(path):
    """Open a Windows file handle without traversing a reparse point."""
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    try:
        attributes = wintypes.DWORD()
        get_attributes = kernel32.GetFileInformationByHandleEx
        get_attributes.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        get_attributes.restype = wintypes.BOOL
        # FileAttributeTagInfo is 9; its first DWORD contains the attributes.
        attribute_tag_info = (wintypes.DWORD * 2)()
        if not get_attributes(
            handle,
            _FILE_ATTRIBUTE_TAG_INFO,
            ctypes.byref(attribute_tag_info),
            ctypes.sizeof(attribute_tag_info),
        ):
            raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
        attributes.value = attribute_tag_info[0]
        if attributes.value & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(
                f"Decoded data cache entries cannot be reparse points: {path}"
            )
        if attributes.value & _FILE_ATTRIBUTE_DIRECTORY:
            raise OSError(f"Decoded data cache entries must be regular files: {path}")
        descriptor = getattr(msvcrt, "open_osfhandle")(
            handle, os.O_RDONLY | getattr(os, "O_BINARY")
        )
    except Exception:
        close_handle(handle)
        raise
    return os.fdopen(descriptor, "rb")


def _raw_preload_open_regular(path):
    """Open a regular cache file without following a final-component link."""
    if _IS_WINDOWS:
        return _raw_preload_open_windows(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        flags |= os.O_NOFOLLOW
    except AttributeError as error:
        raise PermissionError(
            "Automatic Raw preload caching requires no-follow file opens"
        ) from error
    descriptor = os.open(path, flags)
    try:
        _raw_preload_fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "rb")


def _raw_preload_path_stat(path):
    """Return the identity token for a path using one validated handle."""
    with _raw_preload_open_regular(path) as file:
        return _raw_preload_fstat(file.fileno())


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
        sources.append(dict(path=str(path), **_raw_preload_path_stat(path)))
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
    if not _IS_WINDOWS:
        _raw_preload_validate_directory(cache_dir)
        cache_dir.chmod(0o700)
        if cache_dir.stat().st_mode & 0o077:
            raise PermissionError(f"Decoded data cache is not private: {cache_dir}")
    return cache_dir


def _raw_preload_validate_directory(cache_dir):
    """Reject cache paths that another local user could replace."""
    user_id = os.geteuid()
    child_stat = cache_dir.lstat()
    if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISDIR(child_stat.st_mode):
        raise OSError(f"Decoded data cache must be a regular directory: {cache_dir}")
    if child_stat.st_uid != user_id:
        raise PermissionError(
            f"Decoded data cache must be owned by the current user: {cache_dir}"
        )
    if child_stat.st_mode & 0o077:
        raise PermissionError(
            f"Decoded data cache must already be private: {cache_dir}"
        )
    for parent in cache_dir.parents:
        parent_stat = parent.lstat()
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            raise PermissionError(
                f"Decoded data cache cannot have a symlink ancestor: {parent}"
            )
        if parent_stat.st_uid not in (0, user_id):
            raise PermissionError(
                "Decoded data cache cannot have an untrusted owner in its "
                f"physical ancestry: {parent}"
            )
        if parent_stat.st_mode & 0o022 and not parent_stat.st_mode & stat.S_ISVTX:
            raise PermissionError(
                "Decoded data cache cannot have an untrusted writable ancestor: "
                f"{parent}"
            )


def _raw_preload_cache_info(raw):
    """Return the managed cache location and expected array description."""
    cache_identity = raw._decoded_cache_identity()
    if cache_identity is None:
        raise ValueError(
            f'preload="auto" is not supported for {type(raw).__name__}; use '
            "preload=True or an explicit memory-map path"
        )
    decoder_abi, decoder_state = cache_identity
    cache_dir = _raw_preload_cache_dir()
    sources = _raw_preload_source_signature(raw)
    dtype = np.dtype(raw._dtype)
    shape = (int(raw.info["nchan"]), int(raw.n_times))
    identity = dict(
        version=_RAW_PRELOAD_CACHE_VERSION,
        mne_version=MNE_VERSION,
        reader=(type(raw).__module__, type(raw).__qualname__),
        decoder_abi=decoder_abi,
        sources=sources,
        decoder_state=decoder_state,
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


def _raw_preload_generation_name(key, token):
    """Return a unique immutable generation basename."""
    return f"{key}.{token}.data"


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
        file_stat = _raw_preload_fstat(file.fileno())
        if file_stat["size"] > 4096:
            raise ValueError("Oversized Raw preload manifest")
        return json.loads(file.read().decode("utf-8"))


def _raw_preload_cache_read(raw, cache_dir, key, sources, shape, dtype):
    """Read and validate one managed decoded-data cache entry."""
    try:
        manifest = _raw_preload_read_manifest(cache_dir, key)
        if set(manifest) != {
            "version",
            "generation",
            "generation_stat",
        }:
            return None
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if manifest[
            "version"
        ] != _RAW_PRELOAD_CACHE_VERSION or not _raw_preload_generation_valid(
            manifest["generation"], key
        ):
            return None
        generation = cache_dir / manifest["generation"]
        with _raw_preload_open_regular(generation) as file:
            generation_stat = _raw_preload_fstat(file.fileno())
            if (
                generation_stat != manifest["generation_stat"]
                or generation_stat["size"] != nbytes
            ):
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


def _raw_preload_fsync_directory(path):
    """Sync publication metadata where directory fsync is supported."""
    if _IS_WINDOWS:
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)}
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _raw_preload_protect_generation(descriptor):
    """Make a generation owner-read-only where file modes support it."""
    if not _IS_WINDOWS:
        os.fchmod(descriptor, 0o400)


def _raw_preload_replace_manifest(source, destination):
    """Atomically replace a manifest, tolerating transient Windows locks."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))
        else:
            return


def _raw_preload_process_alive(process_id):
    """Return whether a local process still owns a cache publication lock."""
    if process_id <= 0:
        return False
    if _IS_WINDOWS:
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        wait = kernel32.WaitForSingleObject
        wait.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        wait.restype = wintypes.DWORD
        close = kernel32.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        handle = open_process(_PROCESS_SYNCHRONIZE, False, process_id)
        if not handle:
            # Access denied is known-alive; unknown failures are also treated
            # conservatively. ERROR_INVALID_PARAMETER is the invalid-PID case.
            return getattr(ctypes, "get_last_error")() != _ERROR_INVALID_PARAMETER
        try:
            result = wait(handle, 0)
            if result == _WAIT_OBJECT_0:
                return False
            return True  # WAIT_TIMEOUT, WAIT_FAILED, or an unknown result
        finally:
            close(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except (OSError, PermissionError):
        return True
    return True


def _raw_preload_try_lock(lock_path, owner):
    """Try to claim a cache lock without blocking."""
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(owner)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return True


def _raw_preload_remove_stale_lock(lock_path):
    """Remove a dead or abandoned cache lock without touching its target."""
    try:
        lock_stat = lock_path.lstat()
        if lock_path.is_symlink():
            stale = True
        else:
            age = time.time() - lock_stat.st_mtime
            try:
                content = lock_path.read_text(encoding="ascii")
                if len(content) > 256:
                    raise ValueError
                process_id = int(content.split()[0])
                alive = _raw_preload_process_alive(process_id)
            except (IndexError, OSError, OverflowError, ValueError):
                stale = age > 5.0
            else:
                stale = not alive
        current = lock_path.lstat()
        unchanged = (current.st_dev, current.st_ino, current.st_mtime_ns) == (
            lock_stat.st_dev,
            lock_stat.st_ino,
            lock_stat.st_mtime_ns,
        )
        if stale and unchanged:
            lock_path.unlink()
            return True
    except FileNotFoundError:
        return True
    return False


def _raw_preload_release_lock(lock_path, owner):
    """Release a cache lock only when it is still owned by this process."""
    try:
        if (
            not lock_path.is_symlink()
            and lock_path.read_text(encoding="ascii") == owner
        ):
            lock_path.unlink()
    except FileNotFoundError:
        pass


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
    generation_name = _raw_preload_generation_name(key, token)
    generation = cache_dir / generation_name
    temporary = cache_dir / f".{generation_name}.tmp"
    manifest_temporary = None
    manifest_published = False
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+b") as file:
            file.truncate(nbytes)
            data_buffer = np.memmap(file, mode="r+", dtype=dtype, shape=shape)
            data = raw._read_segment(data_buffer=data_buffer)
            try:
                data.flush()
            except BaseException:
                try:
                    data._mmap.close()  # memmap private
                except Exception:
                    pass
                raise
            else:
                data._mmap.close()  # memmap private
            del data, data_buffer
            _raw_preload_protect_generation(file.fileno())
            os.fsync(file.fileno())
        if _raw_preload_source_signature(raw) != sources:
            raise RuntimeError(
                "Source data changed while decoded cache was created; retry"
            )
        os.replace(temporary, generation)
        _raw_preload_fsync_directory(cache_dir)
        manifest = dict(
            version=_RAW_PRELOAD_CACHE_VERSION,
            generation=generation_name,
            generation_stat=_raw_preload_path_stat(generation),
        )
        manifest_temporary = cache_dir / f".{key}.{os.urandom(16).hex()}.json.tmp"
        descriptor = os.open(
            manifest_temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        _raw_preload_replace_manifest(manifest_temporary, cache_dir / f"{key}.json")
        manifest_published = True
        _raw_preload_fsync_directory(cache_dir)
        with _raw_preload_open_regular(generation) as file:
            if _raw_preload_fstat(file.fileno()) != manifest["generation_stat"]:
                raise RuntimeError(
                    "Decoded cache generation changed during publication"
                )
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
    owner = f"{os.getpid()} {os.urandom(16).hex()}"
    deadline = time.monotonic() + _RAW_PRELOAD_LOCK_TIMEOUT
    while True:
        data = _raw_preload_cache_read(raw, cache_dir, key, sources, shape, dtype)
        if data is not None:
            return data
        if _raw_preload_try_lock(key_lock, owner):
            try:
                _raw_preload_scavenge_key(cache_dir, key)
                data = _raw_preload_cache_read(
                    raw, cache_dir, key, sources, shape, dtype
                )
                if data is None:
                    logger.info(f"Creating decoded data cache in {cache_dir}")
                    data = _raw_preload_cache_create(
                        raw, cache_dir, key, sources, shape, dtype
                    )
                _raw_preload_scavenge_key(cache_dir, key)
                return data
            finally:
                _raw_preload_release_lock(key_lock, owner)
        elif _raw_preload_remove_stale_lock(key_lock):
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for decoded cache lock {key_lock}")
        time.sleep(0.025)
