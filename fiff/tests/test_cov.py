import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis-cov.fif')

def test_io_cov():
    """Test IO for noise covariance matrices
    """
    fid, tree, _ = fiff.fiff_open(fname)
    cov_type = 1
    cov = fiff.read_cov(fid, tree, cov_type)
    fid.close()

    fiff.write_cov_file('cov.fif', cov)

    fid, tree, _ = fiff.fiff_open('cov.fif')
    cov2 = fiff.read_cov(fid, tree, cov_type)
    fid.close()

    print assert_array_almost_equal(cov['data'], cov2['data'])