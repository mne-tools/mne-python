# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op

from mne.io import read_raw_nirx
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data
from ....utils import logger


@requires_testing_data
def test_nirx():
    """Test reading NIRX files."""
    fname = op.join(data_path(), 'nirx', 'test_nirx.nxe')
    fname = '/home/rluke/Documents/Repositories/mne-python/mne/io' + \
            '/nirx/tests/nirx_15_2_recording'
    logger.info('Calling loader on %s' % fname)
    raw = read_raw_nirx(fname, preload=True)
    assert raw._data.shape == (288, 16069)
    assert raw.info['subject_info']['sex'] == 2

run_tests_if_main()
