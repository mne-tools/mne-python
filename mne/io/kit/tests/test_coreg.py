# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import inspect
import os

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.io.kit import read_hsp, write_hsp, read_mrk, write_mrk
from mne.utils import _TempDir


FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')
hsp_fname = os.path.join(data_dir, 'test_hsp.txt')
mrk_fname = os.path.join(data_dir, 'test_mrk.sqd')


def test_io_hsp():
    """Test IO for hsp files"""
    tempdir = _TempDir()
    pts = read_hsp(hsp_fname)

    dest = os.path.join(tempdir, 'test.txt')
    write_hsp(dest, pts)
    pts1 = read_hsp(dest)
    assert_array_equal(pts, pts1, "Hsp points diverged after writing and "
                       "reading.")


def test_io_mrk():
    """Test IO for mrk files"""
    tempdir = _TempDir()
    pts = read_mrk(mrk_fname)

    # pickle
    path = os.path.join(tempdir, "mrk.pickled")
    write_mrk(path, pts)
    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write with pickle")

    # txt
    path = os.path.join(tempdir, 'mrk.txt')
    write_mrk(path, pts)
    pts_2 = read_mrk(path)
    assert_array_equal(pts, pts_2, "read/write mrk to text")
