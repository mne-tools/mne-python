"""Run tests for the utilities."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne._fiff import utils as fiff_utils
from mne._fiff.utils import _check_orig_units, _read_segments_file


def test_check_orig_units():
    """Test the checking of original units."""
    orig_units = dict(FC1="nV", Hfp3erz="n/a", Pz="uV", greekMu="μV", microSign="µV")
    orig_units = _check_orig_units(orig_units)
    assert orig_units["FC1"] == "nV"
    assert orig_units["Hfp3erz"] == "n/a"
    assert orig_units["Pz"] == "µV"
    assert orig_units["greekMu"] == "µV"
    assert orig_units["microSign"] == "µV"


@pytest.mark.parametrize("use_mult", (False, True))
def test_read_segments_file_max_block_bytes(tmp_path, use_mult):
    """Test reading in configurable complete channel frames."""
    source = np.arange(20, dtype="<i2").reshape(2, 10, order="F")
    data_fname = tmp_path / "interleaved.bin"
    source.ravel(order="F").tofile(data_fname)
    raw = SimpleNamespace(
        filenames=[data_fname], _raw_extras=[dict(orig_nchan=source.shape[0])]
    )
    if use_mult:
        data = np.empty((1, 7))
        cals = None
        mult = np.array([[0.5, -2.0]])
        want = mult @ source[:, 1:8]
    else:
        data = np.empty((2, 7))
        cals = np.array([0.5, -2.0])
        mult = None
        want = source[:, 1:8] * cals[:, np.newaxis]

    _read_segments_file(
        raw,
        data,
        slice(None),
        0,
        1,
        8,
        cals,
        mult,
        dtype=source.dtype,
        max_block_bytes=1,
    )

    if use_mult:
        assert_allclose(data, want, rtol=1e-15)
    else:
        assert_array_equal(data, want)


def test_read_segments_file_mmap_threaded(tmp_path, monkeypatch):
    """Test mapped blocks are calibrated on worker threads."""
    source = np.arange(40, dtype="<i2").reshape(2, 20, order="F")
    data_fname = tmp_path / "interleaved.bin"
    source.ravel(order="F").tofile(data_fname)
    raw = SimpleNamespace(filenames=[data_fname], _raw_extras=[dict(orig_nchan=2)])
    data = np.empty(source.shape)
    thread_ids = set()
    calibrate = fiff_utils._mult_cal_one

    def _record_thread(*args):
        thread_ids.add(threading.get_ident())
        return calibrate(*args)

    def _fail_fromfile(*args, **kwargs):
        raise AssertionError("mapped reads must not use np.fromfile")

    monkeypatch.setattr(fiff_utils, "_READ_SEGMENTS_FILE_THREAD_MIN_BYTES", 0)
    monkeypatch.setattr(fiff_utils, "_mult_cal_one", _record_thread)
    monkeypatch.setattr(np, "fromfile", _fail_fromfile)
    _read_segments_file(
        raw,
        data,
        slice(None),
        0,
        0,
        source.shape[1],
        np.ones(source.shape[0]),
        None,
        dtype=source.dtype,
        max_block_bytes=8,
        use_mmap=True,
        n_jobs=2,
    )

    assert threading.get_ident() not in thread_ids
    assert_array_equal(data, source)


def test_read_segments_file_mmap_fallback(tmp_path, monkeypatch):
    """Test a failed source mapping falls back to ordinary file reads."""
    source = np.arange(20, dtype="<i2").reshape(2, 10, order="F")
    data_fname = tmp_path / "interleaved.bin"
    source.ravel(order="F").tofile(data_fname)
    raw = SimpleNamespace(filenames=[data_fname], _raw_extras=[dict(orig_nchan=2)])
    data = np.empty(source.shape)

    def _fail_mmap(*args, **kwargs):
        raise OSError("mapping unavailable")

    monkeypatch.setattr(fiff_utils.mmap, "mmap", _fail_mmap)
    _read_segments_file(
        raw,
        data,
        slice(None),
        0,
        0,
        source.shape[1],
        np.ones(source.shape[0]),
        None,
        dtype=source.dtype,
        max_block_bytes=1,
        use_mmap=True,
        n_jobs=2,
    )

    assert_array_equal(data, source)


def test_read_segments_file_mmap_worker_error(tmp_path, monkeypatch):
    """Test worker errors do not prevent closing the source mapping."""
    source = np.arange(20, dtype="<i2").reshape(2, 10, order="F")
    data_fname = tmp_path / "interleaved.bin"
    source.ravel(order="F").tofile(data_fname)
    raw = SimpleNamespace(filenames=[data_fname], _raw_extras=[dict(orig_nchan=2)])
    mappings = []
    map_file = fiff_utils.mmap.mmap

    def _record_mapping(*args, **kwargs):
        mapping = map_file(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    def _fail_calibration(*args):
        raise RuntimeError("expected worker failure")

    monkeypatch.setattr(fiff_utils, "_READ_SEGMENTS_FILE_THREAD_MIN_BYTES", 0)
    monkeypatch.setattr(fiff_utils, "_mult_cal_one", _fail_calibration)
    monkeypatch.setattr(fiff_utils.mmap, "mmap", _record_mapping)
    with pytest.raises(RuntimeError, match="expected worker failure"):
        _read_segments_file(
            raw,
            np.empty(source.shape),
            slice(None),
            0,
            0,
            source.shape[1],
            np.ones(source.shape[0]),
            None,
            dtype=source.dtype,
            max_block_bytes=8,
            use_mmap=True,
            n_jobs=2,
        )

    assert mappings[0].closed
