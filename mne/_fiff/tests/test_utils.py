"""Run tests for the utilities."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

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
