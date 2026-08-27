"""Tests for persistent Raw preload caching."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
import hashlib
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mne
from mne._fiff.pick import pick_info
from mne.io import RawArray, _preload_cache
from mne.io.tests.test_raw import _RawArange, _read_raw_arange

_ORIGINAL_CACHE_REPLACE = None
_IO_DATA_DIR = Path(mne.io.__file__).parent


def _auto_preload_process(reader_name, source, cache_dir):
    """Read one cache entry in an isolated process."""
    os.environ["MNE_CACHE_DIR"] = cache_dir
    raw = getattr(mne.io, reader_name)(source, preload="auto", verbose="error")
    digest = hashlib.sha256(raw.get_data().tobytes()).hexdigest()
    return raw._data.mode, str(raw._data.filename), digest


def _replace_cache_generation_then_exit(source, destination):
    """Crash after publishing data but before publishing its manifest."""
    _ORIGINAL_CACHE_REPLACE(source, destination)
    if str(destination).endswith(".data"):
        os._exit(91)


def _auto_preload_crash_process(source, cache_dir):
    """Run the simulated crash in an isolated process."""
    global _ORIGINAL_CACHE_REPLACE

    os.environ["MNE_CACHE_DIR"] = cache_dir
    _ORIGINAL_CACHE_REPLACE = _preload_cache.os.replace
    _preload_cache.os.replace = _replace_cache_generation_then_exit
    mne.io.read_raw_edf(source, preload="auto", verbose="error")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Configure an isolated cache directory."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("MNE_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def auto_cache(tmp_path, cache_root):
    """Create one stable source file."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"source identity")
    return source, cache_root


def test_auto_preload_api(tmp_path, monkeypatch):
    """Test cache configuration and the literal-path escape."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"source identity")
    monkeypatch.setattr(_preload_cache, "get_config", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="set_cache_dir"):
        _RawArange(preload="auto", filename=source)

    monkeypatch.setattr(
        _preload_cache, "get_config", lambda *args, **kwargs: str(tmp_path)
    )
    with chdir(tmp_path):
        literal = _RawArange(preload=Path("auto"), filename=source)
    assert literal._data.mode == "w+"
    assert (tmp_path / "auto").is_file()

    lazy = _RawArange(preload=False, filename=source)
    lazy.load_data(memmap="auto")
    assert lazy._data.mode == "c"


@pytest.mark.parametrize(
    ("reader_name", "relative_path"),
    (
        ("read_raw_fif", "tests/data/test_raw.fif"),
        ("read_raw_edf", "edf/tests/data/test.edf"),
        ("read_raw_bdf", "edf/tests/data/test.bdf"),
        ("read_raw_brainvision", "brainvision/tests/data/test.vhdr"),
    ),
)
def test_auto_preload_formats(reader_name, relative_path, cache_root):
    """Test exact copy-on-write cache reuse across file formats."""
    source = _IO_DATA_DIR / relative_path
    reader = getattr(mne.io, reader_name)
    expected = reader(source, preload=True, verbose="error").get_data()
    raw = reader(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)
    assert raw._data.mode == "c"
    assert_array_equal(raw.get_data(), expected)
    raw._data[0, 0] += 1.0
    del raw
    gc.collect()

    other = reader(source, preload="auto", verbose="error")
    assert Path(other._data.filename) == generation
    assert_array_equal(other.get_data(), expected)


def test_auto_preload_key(tmp_path, cache_root):
    """Test numeric options and source modification invalidate the cache."""
    data_dir = _IO_DATA_DIR / "brainvision/tests/data"
    for name in ("test.vhdr", "test.vmrk", "test.eeg"):
        shutil.copy(data_dir / name, tmp_path / name)
    source = tmp_path / "test.vhdr"
    raw = mne.io.read_raw_brainvision(source, preload="auto", verbose="error")
    scaled = mne.io.read_raw_brainvision(
        source, scale=2.0, preload="auto", verbose="error"
    )
    assert Path(scaled._data.filename) != Path(raw._data.filename)
    assert_array_equal(scaled.get_data(), 2.0 * raw.get_data())

    text = source.read_text(encoding="utf-8")
    source.write_text(
        text.replace("DataOrientation=MULTIPLEXED", "DataOrientation=VECTORIZED"),
        encoding="utf-8",
    )
    expected = mne.io.read_raw_brainvision(source, preload=True, verbose="error")
    changed = mne.io.read_raw_brainvision(source, preload="auto", verbose="error")
    assert Path(changed._data.filename) != Path(raw._data.filename)
    assert_array_equal(changed.get_data(), expected.get_data())

    source = tmp_path / "copy.edf"
    shutil.copy(_IO_DATA_DIR / "edf/tests/data/test.edf", source)
    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)
    result = source.stat()
    os.utime(source, ns=(result.st_atime_ns, result.st_mtime_ns + 1_000_000_000))
    other = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    assert Path(other._data.filename) != generation


@pytest.mark.parametrize("corruption", ("manifest", "data"))
def test_auto_preload_recovers_corruption(corruption, auto_cache):
    """Test that malformed cache entries become misses."""
    source, cache_root = auto_cache
    raw = _RawArange(preload="auto", filename=source)
    expected = raw.get_data()
    generation = Path(raw._data.filename)
    del raw
    gc.collect()
    cache_dir = next(cache_root.iterdir())
    if corruption == "manifest":
        next(cache_dir.glob("*.json")).write_text("{", encoding="utf-8")
    else:
        generation.write_bytes(b"short")

    other = _RawArange(preload="auto", filename=source)
    assert Path(other._data.filename) != generation
    assert_array_equal(other.get_data(), expected)


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

    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    expected = mne.io.read_raw_edf(source, preload=True, verbose="error").get_data()
    assert_array_equal(raw.get_data(), expected)
    cache_dir = next(cache_root.iterdir())
    assert len(list(cache_dir.glob("*.data"))) == 1
    assert not list(cache_dir.glob("*.tmp"))


def test_auto_preload_rejects_cache_symlink(tmp_path, cache_root):
    """Test that the managed cache directory cannot be redirected."""
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


def test_add_channels_copy_on_write_memmap(tmp_path, monkeypatch):
    """Test adding channels to a copy-on-write memmap."""
    from mne.channels import channels as channels_module

    memmap_fname = tmp_path / "raw-copy-on-write-memmap.dat"
    raw = _read_raw_arange(preload=memmap_fname)
    shape = raw._data.shape
    raw._data._mmap.close()
    raw._data = np.memmap(memmap_fname, mode="c", dtype=np.float64, shape=shape)
    raw._data[0, 0] = 99.0

    info = pick_info(raw.info, [0])
    mne.rename_channels(info, {info["ch_names"][0]: "extra"})
    extra = RawArray(np.zeros((1, raw.n_times)), info)
    monkeypatch.setattr(channels_module.sys, "platform", "linux")
    raw.add_channels([extra])

    assert raw._data.shape == (shape[0] + 1, shape[1])
    assert raw._data[0, 0] == 99.0
    stored = np.memmap(memmap_fname, mode="r", dtype=np.float64, shape=shape)
    assert stored[0, 0] != 99.0
    stored._mmap.close()
