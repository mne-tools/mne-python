# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import inspect
import os

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.io.kit import read_hsp, write_hsp, read_mrk, write_mrk
from mne.coreg import get_ras_to_neuromag_trans
from mne.transforms import apply_trans, rotation, translation
from mne.utils import _TempDir


FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')
hsp_fname = os.path.join(data_dir, 'test_hsp.txt')
mrk_fname = os.path.join(data_dir, 'test_mrk.sqd')
tempdir = _TempDir()


def test_io_hsp():
    """Test IO for hsp files"""
    pts = read_hsp(hsp_fname)

    dest = os.path.join(tempdir, 'test.txt')
    write_hsp(dest, pts)
    pts1 = read_hsp(dest)
    assert_array_equal(pts, pts1, "Hsp points diverged after writing and "
                       "reading.")


def test_io_mrk():
    """Test IO for mrk files"""
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


def test_hsp_trans():
    """Test the coordinate transformation for hsp files"""
    # create model points in neuromag-like space
    anterior = [0, 1, 0]
    left = [-1, 0, 0]
    right = [.8, 0, 0]
    up = [0, 0, 1]
    rand_pts = np.random.uniform(-1, 1, (3, 3))
    pts = np.vstack((anterior, left, right, up, rand_pts))

    # change coord system
    rx, ry, rz, tx, ty, tz = np.random.uniform(-2 * np.pi, 2 * np.pi, 6)
    trans = np.dot(translation(tx, ty, tz), rotation(rx, ry, rz))
    pts_changed = apply_trans(trans, pts)

    # transform back into original space
    nas, lpa, rpa = pts_changed[:3]
    hsp_trans = get_ras_to_neuromag_trans(nas, lpa, rpa)
    pts_restored = apply_trans(hsp_trans, pts_changed)

    assert_array_almost_equal(pts_restored, pts, 6, "Neuromag transformation "
                              "failed")
