# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD-3-Clause

import pickle
from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

from mne.io.kit import read_mrk
from mne.io._digitization import _write_dig_points

mrk_fname = Path(__file__).parent / "data" / "test_mrk.sqd"


def test_io_mrk(tmp_path):
    """Test IO for mrk files."""
    pts = read_mrk(mrk_fname)

    # txt
    path = tmp_path / "mrk.txt"
    _write_dig_points(path, pts)
    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write mrk to text")

    # pickle
    fname = tmp_path / "mrk.pickled"
    with open(fname, "wb") as fid:
        pickle.dump(dict(mrk=pts), fid)
    pts_2 = read_mrk(fname)
    assert_array_equal(pts_2, pts, "pickle mrk")
    with open(fname, "wb") as fid:
        pickle.dump(dict(), fid)
    pytest.raises(ValueError, read_mrk, fname)

    # unsupported extension
    pytest.raises(ValueError, read_mrk, "file.ext")
