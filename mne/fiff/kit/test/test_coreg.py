# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import inspect
import os

from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.fiff.kit.coreg import read_hsp, write_hsp
from mne.utils import _TempDir

FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')
hsp_fname = os.path.join(data_dir, 'test_hsp.txt')

tempdir = _TempDir()


def test_io_hsp():
    """Test IO for hsp files"""
    pts = read_hsp(hsp_fname)

    dest = os.path.join(tempdir, 'test.txt')
    write_hsp(dest, pts)
    pts1 = read_hsp(dest)

    assert_array_equal(pts, pts1)
