"""Tests for persistent Raw preload caching."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
import hashlib
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
from mne.io.tests.test_raw import _read_raw_arange

_IO_DATA_DIR = Path(mne.io.__file__).parent


def _auto_preload_process(reader_name, source, cache_dir):
    """Read one cache entry in an isolated process."""
    os.environ["MNE_CACHE_DIR"] = cache_dir
    raw = getattr(mne.io, reader_name)(source, preload="auto", verbose="error")
    digest = hashlib.sha256(raw.get_data().tobytes()).hexdigest()
    return raw._data.mode, str(raw._data.filename), digest


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Configure an isolated cache directory."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("MNE_CACHE_DIR", str(cache_root))
    return cache_root


def test_auto_preload_api(tmp_path, monkeypatch):
    """Test cache configuration and the literal-path escape."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    monkeypatch.setattr(_preload_cache, "get_config", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="set_cache_dir"):
        mne.io.read_raw_edf(source, preload="auto", verbose="error")

    with chdir(tmp_path):
        literal = mne.io.read_raw_edf(source, preload=Path("auto"), verbose="error")
    assert literal._data.mode == "w+"
    assert (tmp_path / "auto").is_file()

    # load_data(memmap="auto") resolves the same sentinel as preload="auto"
    lazy = mne.io.read_raw_edf(source, preload=False, verbose="error")
    with pytest.raises(ValueError, match="set_cache_dir"):
        lazy.load_data(memmap="auto")


@pytest.mark.parametrize("fname", ("test_raw.fif", "test_raw.fif.gz"))
def test_auto_preload_fif(fname, cache_root):
    """Test cache reuse for FIF, whose reader tests skip test_preloading."""
    source = _IO_DATA_DIR / "tests/data" / fname
    expected = mne.io.read_raw_fif(source, preload=True, verbose="error").get_data()
    raw = mne.io.read_raw_fif(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)
    assert raw._data.mode == "c"
    assert_array_equal(raw.get_data(), expected)
    del raw
    gc.collect()

    other = mne.io.read_raw_fif(source, preload="auto", verbose="error")
    assert Path(other._data.filename) == generation
    assert_array_equal(other.get_data(), expected)


def test_auto_preload_identity(tmp_path, cache_root):
    """Test reader options and source modification invalidate the cache."""
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

    source = tmp_path / "copy.edf"
    shutil.copy(_IO_DATA_DIR / "edf/tests/data/test.edf", source)
    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    generation = Path(raw._data.filename)
    result = source.stat()
    os.utime(source, ns=(result.st_atime_ns, result.st_mtime_ns + 1_000_000_000))
    other = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    assert Path(other._data.filename) != generation


def test_auto_preload_recovers_corruption(cache_root):
    """Test that a truncated deterministic cache entry is rebuilt."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    raw = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    expected = raw.get_data().copy()
    generation = Path(raw._data.filename)
    del raw
    gc.collect()
    generation.write_bytes(b"short")

    other = mne.io.read_raw_edf(source, preload="auto", verbose="error")
    assert Path(other._data.filename) == generation
    assert_array_equal(other.get_data(), expected)


def test_auto_preload_concurrent_misses(cache_root):
    """Test that concurrent misses publish one exact cache entry."""
    source = _IO_DATA_DIR / "edf/tests/data/test.edf"
    args = ("read_raw_edf", str(source), str(cache_root))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_auto_preload_process, *zip(*(args,) * 4)))

    assert {result[0] for result in results} == {"c"}
    assert len({result[1] for result in results}) == 1
    assert len({result[2] for result in results}) == 1


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
