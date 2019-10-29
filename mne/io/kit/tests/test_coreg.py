# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import inspect
import os
import pickle

import pytest
from numpy.testing import assert_array_equal

from mne.io.kit import read_mrk
from mne.io._digitization import _write_dig_points
from mne.utils import _TempDir


FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')
mrk_fname = os.path.join(data_dir, 'test_mrk.sqd')


def test_io_mrk():
    """Test IO for mrk files."""
    tempdir = _TempDir()
    pts = read_mrk(mrk_fname)

    # txt
    path = os.path.join(tempdir, 'mrk.txt')
    _write_dig_points(path, pts)
    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write mrk to text")

    # pickle
    fname = os.path.join(tempdir, 'mrk.pickled')
    with open(fname, 'wb') as fid:
        pickle.dump(dict(mrk=pts), fid)
    pts_2 = read_mrk(fname)
    assert_array_equal(pts_2, pts, "pickle mrk")
    with open(fname, 'wb') as fid:
        pickle.dump(dict(), fid)
    pytest.raises(ValueError, read_mrk, fname)

    # unsupported extension
    pytest.raises(ValueError, read_mrk, "file.ext")
