import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

import fiff

MNE_SAMPLE_DATASET_PATH = os.getenv('MNE_SAMPLE_DATASET_PATH')
fname = op.join(MNE_SAMPLE_DATASET_PATH, 'MEG', 'sample',
                                            'sample_audvis_raw.fif')

def test_io_raw():
    """Test IO for raw data
    """
    raw = fiff.setup_read_raw(fname)

    nchan = raw['info']['nchan']
    ch_names = raw['info']['ch_names']
    meg_channels_idx = [k for k in range(nchan) if ch_names[k][:3]=='MEG']
    meg_channels_idx = meg_channels_idx[:5]

    data, times = fiff.read_raw_segment_times(raw, from_=100, to=115,
                                              sel=meg_channels_idx)

