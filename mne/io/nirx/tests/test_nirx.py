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
    fname = op.join(data_path(), 'NIRx', 'nirx_15_2_recording_w_short')
    logger.info('Calling loader on %s' % fname)
    raw = read_raw_nirx(fname, preload=True)
    assert raw._data.shape == (90, 144)
    assert raw.info['subject_info']['sex'] == 1

run_tests_if_main()
