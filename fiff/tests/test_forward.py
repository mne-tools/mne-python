import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis-ave-7-fwd.fif')

def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = fiff.read_forward_solution(fname)
    fwd = fiff.read_forward_solution(fname, force_fixed=True)
    leadfield = fwd['sol']['data']
