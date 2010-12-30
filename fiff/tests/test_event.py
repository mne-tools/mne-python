import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis_raw-eve.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    event_list = fiff.read_events(fname)
    fiff.write_events('events.fif', event_list)
    event_list2 = fiff.read_events(fname)
    assert_array_almost_equal(event_list, event_list2)
