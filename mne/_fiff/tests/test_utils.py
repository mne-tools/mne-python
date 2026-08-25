"""Run tests for the utilities."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pickle
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from mne._fiff.utils import _check_orig_units, _memmap_for, _read_segments_file


def test_check_orig_units():
    """Test the checking of original units."""
    orig_units = dict(FC1="nV", Hfp3erz="n/a", Pz="uV", greekMu="μV", microSign="µV")
    orig_units = _check_orig_units(orig_units)
    assert orig_units["FC1"] == "nV"
    assert orig_units["Hfp3erz"] == "n/a"
    assert orig_units["Pz"] == "µV"
    assert orig_units["greekMu"] == "µV"
    assert orig_units["microSign"] == "µV"


def test_read_segments_file_unaligned_offset(tmp_path):
    """Test reading multi-byte data at an unaligned byte offset."""
    fname = tmp_path / "test.bin"
    values = np.array([[1, 2, 3], [10, 20, 30]], dtype="<i2")
    fname.write_bytes(b"x" + values.T.tobytes() + b"x")
    raw = SimpleNamespace(
        _raw_extras=[dict(orig_nchan=2)],
        filenames=[fname],
    )
    data = np.empty((2, 3))

    _read_segments_file(
        raw,
        data,
        slice(None),
        0,
        0,
        3,
        np.ones(2),
        None,
        dtype="<i2",
        offset=1,
    )

    assert_array_equal(data, values)


def test_memmap_cache_deepcopy(tmp_path):
    """Test that copying a cache does not copy the mapped file into memory."""
    fname = tmp_path / "test.bin"
    fname.write_bytes(b"test")
    extras = {}
    original = _memmap_for(extras, fname)

    copied = _memmap_for(deepcopy(extras), fname)

    assert copied is not original
    assert copied.filename == str(fname)
    assert not copied.flags.writeable


def test_memmap_cache_pickle(tmp_path):
    """Test that pickling a cache does not serialize the mapped file."""
    fname = tmp_path / "test.bin"
    fname.write_bytes(b"test")
    extras = {}
    original = _memmap_for(extras, fname)

    copied = _memmap_for(pickle.loads(pickle.dumps(extras)), fname)

    assert copied is not original
    assert copied.filename == str(fname)
    assert not copied.flags.writeable


def test_memmap_cache_failed_pid_change(tmp_path, monkeypatch):
    """Test that a failed process-local remap releases inherited state."""
    fname = tmp_path / "test.bin"
    fname.write_bytes(b"test")
    extras = {}
    original = _memmap_for(extras, fname)
    cache = extras["_memmap_cache"]

    monkeypatch.setattr("mne._fiff.utils.os.getpid", lambda: cache.pid + 1)
    monkeypatch.setattr(np, "memmap", Mock(side_effect=OSError))

    assert _memmap_for(extras, fname) is None
    assert original._mmap.closed
    assert cache.mapping is None
    assert cache.pid is None
