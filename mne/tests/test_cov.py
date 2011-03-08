import os.path as op

from numpy.testing import assert_array_almost_equal

import mne
from ..fiff import fiff_open

fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data', 'test-cov.fif')

def test_io_cov():
    """Test IO for noise covariance matrices
    """
    fid, tree, _ = fiff_open(fname)
    cov_type = 1
    cov = mne.read_cov(fid, tree, cov_type)
    fid.close()

    mne.write_cov_file('cov.fif', cov)

    fid, tree, _ = fiff_open('cov.fif')
    cov2 = mne.read_cov(fid, tree, cov_type)
    fid.close()

    print assert_array_almost_equal(cov['data'], cov2['data'])