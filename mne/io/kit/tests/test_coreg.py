# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import inspect
import os

from numpy.testing import assert_array_equal

from mne.io.kit import read_mrk
from mne.io.meas_info import _write_dig_points
from mne.utils import _TempDir


FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')
mrk_fname = os.path.join(data_dir, 'test_mrk.sqd')


def test_io_mrk():
    """Test IO for mrk files"""
    tempdir = _TempDir()
    pts = read_mrk(mrk_fname)

    # txt
    path = os.path.join(tempdir, 'mrk.txt')
    _write_dig_points(path, pts)
    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write mrk to text")
