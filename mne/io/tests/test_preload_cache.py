"""Tests for persistent Raw preload-cache infrastructure."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mne
from mne._fiff.pick import pick_info
from mne.io import RawArray, _preload_cache
from mne.io.tests.test_raw import (
    _RawArange,
    _read_raw_arange,
)

_ORIGINAL_CACHE_REPLACE = None
_IO_DATA_DIR = Path(mne.io.__file__).parent


def _auto_preload_process(reader_name, source, cache_dir):
    """Read one automatic cache entry in an isolated process."""
    os.environ["MNE_CACHE_DIR"] = cache_dir
    raw = getattr(mne.io, reader_name)(source, preload="auto", verbose="error")
    digest = hashlib.sha256(raw.get_data().tobytes()).hexdigest()
    return raw._data.mode, str(raw._data.filename), digest


def _replace_cache_generation_then_exit(source, destination):
    """Publish one generation and simulate an immediate process crash."""
    _ORIGINAL_CACHE_REPLACE(source, destination)
    if str(destination).endswith(".data"):
        os._exit(91)


def _auto_preload_crash_process(source, cache_dir):
    """Crash a cache writer after its generation becomes durable."""
    global _ORIGINAL_CACHE_REPLACE

    os.environ["MNE_CACHE_DIR"] = cache_dir
    _ORIGINAL_CACHE_REPLACE = _preload_cache.os.replace
    _preload_cache.os.replace = _replace_cache_generation_then_exit
    mne.io.read_raw_edf(source, preload="auto", verbose="error")


class _RawArangeBarrier(_RawArange):
    _barrier = None

    def _read_segment(self, *args, **kwargs):
        self._barrier.wait(timeout=2.0)
        return super()._read_segment(*args, **kwargs)


class _RawArangeRecording(_RawArange):
    _mappings = None

    def _read_segment(self, *args, **kwargs):
        data = super()._read_segment(*args, **kwargs)
        self._mappings.append(data)
        return data


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Configure and return an isolated preload cache."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("MNE_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def auto_cache(tmp_path, cache_root):
    """Create one stable source for the isolated preload cache."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"source identity")
    return source, cache_root


def _fail_manifest_dump(*args, **kwargs):
    raise OSError("injected manifest failure")


def _fail_memmap_flush(self):
    raise OSError("injected flush failure")


def test_unlocked_cache_lock_never_blocks(auto_cache, monkeypatch):
    """Test that an abandoned unlocked file cannot block cache creation."""
    source, _ = auto_cache
    raw = _RawArange(preload=False, filename=source)
    cache_dir, key, _, _, _ = _preload_cache._raw_preload_cache_info(raw)
    (cache_dir / f"{key}.lock").write_text("abandoned", encoding="ascii")
    monkeypatch.setattr(_preload_cache, "_RAW_PRELOAD_LOCK_TIMEOUT", 0.1)

    raw.load_data(memmap="auto")

    assert_array_equal(raw.get_data()[:, 0], np.arange(1, 9))


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_cache_accepts_configured_shared_ancestor(tmp_path):
    """Test that the explicitly configured cache location is trusted."""
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    cache_root = shared / "cache"
    cache_root.mkdir(mode=0o700)
    assert _preload_cache._raw_preload_cache_dir(cache_root).is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_cache_accepts_configured_public_managed_directory(tmp_path):
    """Test that an explicitly configured existing cache is trusted."""
    managed = tmp_path / "raw-preload-v1"
    managed.mkdir(mode=0o777)
    managed.chmod(0o777)
    assert _preload_cache._raw_preload_cache_dir(tmp_path) == managed
    assert managed.stat().st_mode & 0o777 == 0o777


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_cache_canonicalizes_symlink_ancestor(tmp_path):
    """Test that a configured symlink is fixed to one physical location."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    managed = _preload_cache._raw_preload_cache_dir(link)
    assert managed == real / "raw-preload-v1"


def test_distinct_keys_publish_concurrently(tmp_path, cache_root, monkeypatch):
    """Test that unrelated first-time decodes do not share a long-held lock."""
    sources = [tmp_path / f"source-{index}.bin" for index in range(2)]
    for index, source in enumerate(sources):
        source.write_bytes(bytes([index]))
    raws = [_RawArangeBarrier(preload=False, filename=source) for source in sources]
    barrier = threading.Barrier(2)
    monkeypatch.setattr(_RawArangeBarrier, "_barrier", barrier)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda raw: raw.load_data(memmap="auto"), raws))
    assert len({Path(raw._data.filename) for raw in raws}) == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX replacement semantics")
def test_generation_mapping_uses_validated_handle(tmp_path):
    """Test that pathname replacement cannot change the mapped generation."""
    generation = tmp_path / "generation.data"
    replacement = tmp_path / "replacement.data"
    np.arange(4.0).tofile(generation)
    np.full(4, 99.0).tofile(replacement)
    with _preload_cache._raw_preload_open_regular(generation) as file:
        os.replace(replacement, generation)
        data = np.memmap(file, mode="c", dtype=np.float64, shape=(4,))
    np.testing.assert_array_equal(data, np.arange(4.0))


def test_auto_preload_first_miss_is_copy_on_write(auto_cache, tmp_path):
    """Test that an automatic cache miss publishes immutable data."""
    source, _ = auto_cache
    with chdir(tmp_path):
        raw = _RawArange(preload="auto", filename=source)

    assert isinstance(raw._data, np.memmap)
    assert raw._data.mode == "c"
    generation = Path(raw._data.filename)
    expected = raw.get_data()
    raw._data[0, 0] = 99.0
    del raw
    gc.collect()

    with chdir(tmp_path):
        other = _RawArange(preload="auto", filename=source)
    assert other._data.mode == "c"
    assert Path(other._data.filename) == generation
    assert_array_equal(other.get_data(), expected)
    assert not (tmp_path / "auto").exists()


def test_auto_preload_ignores_generation_metadata(auto_cache):
    """Test that generation metadata does not cause an expensive re-decode."""
    source, _ = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    generation = Path(raw._data.filename)
    del raw
    gc.collect()
    result = generation.stat()
    os.utime(
        generation,
        ns=(result.st_atime_ns, result.st_mtime_ns + 1_000_000),
    )

    other = _RawArange(preload="auto", filename=source)

    assert Path(other._data.filename) == generation


def test_auto_preload_scavenges_same_key(auto_cache):
    """Test that a retry removes abandoned files for its cache key."""
    source, _ = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    cache_dir = Path(raw._data.filename).parent
    manifest_path = next(cache_dir.glob("*.json"))
    key = manifest_path.stem
    orphan = cache_dir / f"{key}.{'0' * 32}.data"
    temporary = cache_dir / f".{key}.abandoned.tmp"
    orphan.write_bytes(b"orphan")
    temporary.write_bytes(b"temporary")
    manifest_path.unlink()

    other = _RawArange(preload="auto", filename=source)
    assert_array_equal(other.get_data(), raw.get_data())
    assert len(list(cache_dir.glob(f"{key}.*.data"))) == 1
    assert not temporary.exists()


def test_auto_preload_cleans_failed_publication(auto_cache, monkeypatch):
    """Test cleanup and retry after manifest publication fails."""
    source, cache_root = auto_cache

    with monkeypatch.context() as context:
        context.setattr(_preload_cache.json, "dump", _fail_manifest_dump)
        with pytest.raises(OSError, match="injected manifest failure"):
            _RawArange(preload="auto", filename=source)

    cache_dir = cache_root / "raw-preload-v1"
    assert not list(cache_dir.glob("*.data"))
    assert not list(cache_dir.glob("*.json"))
    assert not list(cache_dir.glob("*.tmp"))
    raw = _RawArange(preload="auto", filename=source)
    assert_array_equal(raw.get_data()[:, 0], np.arange(1, 9))


def test_auto_preload_closes_failed_flush(auto_cache, monkeypatch):
    """Test that a flush failure closes its temporary mapping."""
    source, cache_root = auto_cache
    mappings = []
    monkeypatch.setattr(_RawArangeRecording, "_mappings", mappings)
    monkeypatch.setattr(np.memmap, "flush", _fail_memmap_flush)
    with pytest.raises(OSError, match="injected flush failure"):
        _RawArangeRecording(preload="auto", filename=source)

    assert mappings[0]._mmap.closed
    cache_dir = cache_root / "raw-preload-v1"
    assert not list(cache_dir.glob("*.data"))
    assert not list(cache_dir.glob("*.json"))
    assert not list(cache_dir.glob("*.tmp"))


def test_auto_preload_invalidates_mne_version(auto_cache, monkeypatch):
    """Test that decoded cache data do not cross MNE version boundaries."""
    source, _ = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    first = Path(raw._data.filename)

    monkeypatch.setattr(_preload_cache, "MNE_VERSION", "next-version")
    other = _RawArange(preload="auto", filename=source)
    assert Path(other._data.filename) != first
    assert_array_equal(other.get_data(), raw.get_data())


def test_auto_preload_api_contract(tmp_path, monkeypatch):
    """Test automatic preload errors and the literal-path escape."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"source identity")
    monkeypatch.setattr(_preload_cache, "get_config", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="set_cache_dir"):
        _RawArange(preload="auto", filename=source)
    raw = _RawArange(preload=False, filename=source)
    with pytest.raises(ValueError, match="set_cache_dir"):
        raw.load_data(memmap="auto")
    identity_method = _RawArange._decoded_cache_identity
    monkeypatch.setattr(_RawArange, "_decoded_cache_identity", lambda self: None)
    monkeypatch.setattr(
        _preload_cache, "get_config", lambda *args, **kwargs: str(tmp_path)
    )
    with pytest.raises(ValueError, match="is not supported"):
        _RawArange(preload="auto", filename=source)

    with chdir(tmp_path):
        literal = _RawArange(preload=Path("auto"), filename=source)
    assert literal._data.mode == "w+"
    assert (tmp_path / "auto").is_file()

    monkeypatch.setattr(_RawArange, "_decoded_cache_identity", identity_method)
    lazy = _RawArange(preload=False, filename=source)
    lazy.load_data(memmap="auto")
    assert lazy._data.mode == "c"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_auto_preload_storage_permissions(auto_cache):
    """Test that newly created cache files are not shared by default."""
    source, cache_root = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    cache_dir = next(cache_root.iterdir())
    manifest = next(cache_dir.glob("*.json"))
    generation = Path(raw._data.filename)
    assert cache_dir.stat().st_mode & 0o077 == 0
    assert manifest.stat().st_mode & 0o077 == 0
    assert generation.stat().st_mode & 0o077 == 0


def test_auto_preload_rejects_cache_symlink(tmp_path, cache_root):
    """Test that the private managed directory cannot be redirected."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"source identity")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, cache_root / "raw-preload-v1")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(OSError, match="regular directory"):
        _RawArange(preload="auto", filename=source)
    assert not list(outside.iterdir())


@pytest.mark.parametrize(
    ("reader_name", "relative_path"),
    (
        ("read_raw_fif", "tests/data/test_raw.fif"),
        ("read_raw_edf", "edf/tests/data/test.edf"),
        ("read_raw_bdf", "edf/tests/data/test.bdf"),
        ("read_raw_brainvision", "brainvision/tests/data/test.vhdr"),
    ),
)
def test_auto_preload_cache_formats(reader_name, relative_path, cache_root):
    """Test exact automatic preload reuse across supported formats."""
    source = _IO_DATA_DIR / relative_path
    reader = getattr(mne.io, reader_name)
    reference = reader(source, preload=True, verbose="error").get_data()

    raw = reader(source, preload="auto", verbose="error")
    assert raw._data.mode == "c"
    generation = Path(raw._data.filename)
    assert_array_equal(raw.get_data(), reference)
    raw._data[0, 0] += 1.0
    del raw
    gc.collect()

    other = reader(source, preload="auto", verbose="error")
    assert other._data.mode == "c"
    assert Path(other._data.filename) == generation
    assert_array_equal(other.get_data(), reference)


@pytest.mark.parametrize(
    "corruption",
    (
        "truncated_json",
        "oversized_manifest",
        "symlink_manifest",
        "missing_field",
        "unknown_field",
        "traversal_generation",
        "missing_generation",
        "wrong_size_generation",
        "symlink_generation",
    ),
)
def test_auto_preload_cache_corruption(corruption, auto_cache, tmp_path):
    """Test that corrupt cache entries always become safe misses."""
    source, cache_root = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    expected = raw.get_data()
    del raw
    cache_dir = next(cache_root.iterdir())
    manifest_path = next(cache_dir.glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = cache_dir / manifest["generation"]

    if corruption == "truncated_json":
        manifest_path.write_text('{"version":', encoding="utf-8")
    elif corruption == "oversized_manifest":
        manifest_path.write_text(" " * 4097, encoding="utf-8")
    elif corruption == "symlink_manifest":
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_path.unlink()
        try:
            os.symlink(outside, manifest_path)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    elif corruption == "missing_field":
        manifest.pop("generation")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "unknown_field":
        manifest["unknown"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "traversal_generation":
        manifest["generation"] = f"../{manifest['generation']}"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "missing_generation":
        generation.unlink()
    elif corruption == "wrong_size_generation":
        generation.chmod(0o600)
        generation.write_bytes(b"short")
    else:
        outside = tmp_path / "outside.dat"
        outside.write_bytes(b"outside")
        generation.unlink()
        try:
            os.symlink(outside, generation)
        except OSError:
            pytest.skip("symlink creation is unavailable")

    other = _RawArange(preload="auto", filename=source)
    assert other._data.mode == "c"
    assert_array_equal(other.get_data(), expected)
    if corruption == "symlink_generation":
        assert outside.read_bytes() == b"outside"
    elif corruption == "symlink_manifest":
        assert json.loads(outside.read_text(encoding="utf-8")) == manifest


def test_auto_preload_concurrent_misses(cache_root):
    """Test that concurrent misses publish one exact generation."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    args = ("read_raw_edf", str(source), str(cache_root))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_auto_preload_process, *zip(*(args,) * 4)))

    assert {result[0] for result in results} == {"c"}
    assert len({result[1] for result in results}) == 1
    assert len({result[2] for result in results}) == 1
    cache_dir = next(cache_root.iterdir())
    assert len(list(cache_dir.glob("*.data"))) == 1
    assert not list(cache_dir.glob("*.tmp"))


def test_auto_preload_recovers_crashed_publisher(cache_root):
    """Test recovery when a writer dies before manifest publication."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_auto_preload_crash_process, args=(str(source), str(cache_root))
    )
    process.start()
    process.join(timeout=15)
    assert process.exitcode == 91
    cache_dir = next(cache_root.iterdir())
    assert len(list(cache_dir.glob("*.data"))) == 1
    assert not list(cache_dir.glob("*.json"))

    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    reference = mne.io.read_raw_edf(source, preload=True, verbose="error")
    assert raw._data.mode == "c"
    assert_array_equal(raw.get_data(), reference.get_data())
    assert not list(cache_dir.glob("*.tmp"))
    assert len(list(cache_dir.glob("*.data"))) == 1


def test_auto_preload_identity_ignores_edf_channel_type(cache_root):
    """Test that live metadata does not invalidate decoded samples."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)

    other = mne.io.read_raw_edf(source, eog=[0], preload="auto", verbose="error")
    assert Path(other._data.filename) == generation
    assert other.get_channel_types()[0] == "eog"
    assert_array_equal(other.get_data(), raw.get_data())


def test_auto_preload_brainvision_live_markers(tmp_path, cache_root):
    """Test that markers remain live while decoded samples are reused."""
    data_dir = _IO_DATA_DIR / "brainvision/tests/data"
    for name in ("test.vhdr", "test.vmrk", "test.eeg"):
        shutil.copy(data_dir / name, tmp_path / name)
    source = tmp_path / "test.vhdr"
    raw = mne.io.read_raw_brainvision(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)
    annotation_count = len(raw.annotations)
    with (tmp_path / "test.vmrk").open("a", encoding="utf-8") as file:
        file.write("\nMk15=Stimulus,S 99,7800,1,0\n")

    other = mne.io.read_raw_brainvision(source, preload="auto", verbose="error")
    assert Path(other._data.filename) == generation
    assert len(other.annotations) == annotation_count + 1
    assert other.annotations.description[-1] == "Stimulus/S 99"
    plain = mne.io.read_raw_brainvision(
        source, ignore_marker_types=True, preload="auto", verbose="error"
    )
    assert Path(plain._data.filename) == generation
    assert plain.annotations.description[-1] == "S 99"


def test_auto_preload_numeric_invalidation(tmp_path, cache_root):
    """Test numeric options and filesystem changes invalidate cached data."""
    data_dir = _IO_DATA_DIR / "brainvision/tests/data"
    source = data_dir / "test.vhdr"
    raw = mne.io.read_raw_brainvision(source, preload="auto", verbose="error")
    scaled = mne.io.read_raw_brainvision(
        source, scale=2.0, preload="auto", verbose="error"
    )
    assert Path(scaled._data.filename) != Path(raw._data.filename)
    assert_array_equal(scaled.get_data(), 2.0 * raw.get_data())

    edf_source = tmp_path / "test.edf"
    shutil.copy(_IO_DATA_DIR / "edf/tests/data/test.edf", edf_source)
    original = mne.io.read_raw_edf(edf_source, preload="auto", verbose="error")
    generation = Path(original._data.filename)
    excluded = mne.io.read_raw_edf(
        edf_source,
        exclude=[original.ch_names[0]],
        preload="auto",
        verbose="error",
    )
    assert Path(excluded._data.filename) != generation
    assert excluded._data.shape[0] == original._data.shape[0] - 1
    result = edf_source.stat()
    with edf_source.open("r+b") as file:
        file.seek(-1, os.SEEK_END)
        byte = file.read(1)
        file.seek(-1, os.SEEK_END)
        file.write(bytes([byte[0] ^ 1]))
    os.utime(
        edf_source,
        ns=(result.st_atime_ns, result.st_mtime_ns + 10_000_000_000),
    )
    changed = mne.io.read_raw_edf(edf_source, preload="auto", verbose="error")
    assert Path(changed._data.filename) != generation

    alias = tmp_path / "alias.edf"
    try:
        os.symlink(edf_source, alias)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    aliased = mne.io.read_raw_edf(alias, preload="auto", verbose="error")
    assert Path(aliased._data.filename) == Path(changed._data.filename)


@pytest.mark.parametrize("attribute", ("_projector", "_comp"))
def test_auto_preload_transform_invalidation(attribute, auto_cache):
    """Test delayed projection and compensation use distinct cache entries."""
    source, _ = auto_cache
    transformed = _read_raw_arange(filename=source)
    setattr(transformed, attribute, 2.0 * np.eye(len(transformed.ch_names)))
    transformed.load_data(memmap="auto", verbose="error")
    plain = _read_raw_arange(filename=source)
    plain.load_data(memmap="auto", verbose="error")

    assert Path(transformed._data.filename) != Path(plain._data.filename)
    assert_array_equal(transformed.get_data(), 2.0 * plain.get_data())


def test_add_channels_copy_on_write_memmap(tmp_path, monkeypatch):
    """Test adding channels to a copy-on-write memmap."""
    from mne.channels import channels as channels_module

    memmap_fname = tmp_path / "raw-copy-on-write-memmap.dat"
    raw = _read_raw_arange(preload=memmap_fname)
    shape = raw._data.shape
    expected = raw._data.copy()
    raw._data._mmap.close()
    raw._data = np.memmap(memmap_fname, mode="c", dtype=np.float64, shape=shape)
    raw._data[0, 0] = 99.0

    info = pick_info(raw.info, [0])
    mne.rename_channels(info, {info["ch_names"][0]: "extra"})
    extra = RawArray(np.zeros((1, raw.n_times)), info)
    monkeypatch.setattr(channels_module.sys, "platform", "linux")
    raw.add_channels([extra])

    assert not isinstance(raw._data, np.memmap)
    assert raw._data.shape == (shape[0] + 1, shape[1])
    expected[0, 0] = 99.0
    assert_array_equal(raw._data[:-1], expected)
    stored = np.memmap(memmap_fname, mode="r", dtype=np.float64, shape=shape)
    assert stored[0, 0] != 99.0
    stored._mmap.close()
