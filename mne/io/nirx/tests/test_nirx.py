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

    # Test data import
    assert raw._data.shape == (26, 144)
    assert raw.info['sfreq'] == 12.5
    assert raw.info['ch_names'][0] == "S1-D1 760 (nm)"
    assert raw.info['ch_names'][1] == "S1-D1 850 (nm)"
    assert raw.info['ch_names'][2] == "S1-D9 760 (nm)"
    assert raw.info['ch_names'][3] == "S1-D9 850 (nm)"
    assert raw.info['ch_names'][24] == "S5-D13 760 (nm)"
    assert raw.info['ch_names'][25] == "S5-D13 850 (nm)"

    # Test info import
    assert raw.info['subject_info']['sex'] == 1
    assert raw.info['subject_info']['first_name'] == "MNE"
    assert raw.info['subject_info']['middle_name'] == "Test"
    assert raw.info['subject_info']['last_name'] == "Recording"


run_tests_if_main()
