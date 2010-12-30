import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis-ave.fif')

def test_io_cov():
    """Test IO for noise covariance matrices
    """
    data = fiff.read_evoked(fname)

    fiff.write_evoked('evoked.fif', data)
    data2 = fiff.read_evoked('evoked.fif')

    print assert_array_almost_equal(data['evoked']['epochs'],
                                    data2['evoked']['epochs'])
    print assert_array_almost_equal(data['evoked']['times'],
                                    data2['evoked']['times'])
    print assert_equal(data['evoked']['nave'],
                                    data2['evoked']['nave'])
    print assert_equal(data['evoked']['aspect_kind'],
                                    data2['evoked']['aspect_kind'])
    print assert_equal(data['evoked']['last'],
                                    data2['evoked']['last'])
    print assert_equal(data['evoked']['first'],
                                    data2['evoked']['first'])
