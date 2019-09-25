# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license

import os.path as op

import numpy as np

from mne.io import read_raw_nirx
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data

fname_nirx = op.join(data_path(download=False),
                     'NIRx', 'nirx_15_2_recording_w_short')


@requires_testing_data
def test_nirx():
    """Test reading NIRX files."""
    raw = read_raw_nirx(fname_nirx, preload=True)

    # Test data import
    assert raw._data.shape == (26, 145)
    assert raw.info['sfreq'] == 12.5

    # Test channel naming
    assert raw.info['ch_names'][0] == "S1-D1 760"
    assert raw.info['ch_names'][1] == "S1-D1 850"
    assert raw.info['ch_names'][2] == "S1-D9 760"
    assert raw.info['ch_names'][3] == "S1-D9 850"
    assert raw.info['ch_names'][24] == "S5-D13 760"
    assert raw.info['ch_names'][25] == "S5-D13 850"

    # Test info import
    assert raw.info['subject_info']['sex'] == 1
    assert raw.info['subject_info']['first_name'] == "MNE"
    assert raw.info['subject_info']['middle_name'] == "Test"
    assert raw.info['subject_info']['last_name'] == "Recording"

    # Test distance between optodes matches values from
    # nirsite https://github.com/mne-tools/mne-testing-data/pull/51
    # step 4 figure 2
    allowed_distance_error = 0.2
    distances = raw._probe_distances()
    assert abs(distances[0] - 30.4) < allowed_distance_error
    assert abs(distances[2] - 7.8) < allowed_distance_error
    assert abs(distances[4] - 31.0) < allowed_distance_error
    assert abs(distances[6] - 8.6) < allowed_distance_error
    assert abs(distances[8] - 41.6) < allowed_distance_error
    assert abs(distances[10] - 7.2) < allowed_distance_error
    assert abs(distances[12] - 38.9) < allowed_distance_error
    assert abs(distances[14] - 7.5) < allowed_distance_error
    assert abs(distances[16] - 55.8) < allowed_distance_error
    assert abs(distances[18] - 56.2) < allowed_distance_error
    assert abs(distances[20] - 56.1) < allowed_distance_error
    assert abs(distances[22] - 56.5) < allowed_distance_error
    assert abs(distances[24] - 7.7) < allowed_distance_error

    # Test which channels are short
    # These are the ones marked as red at
    # https://github.com/mne-tools/mne-testing-data/pull/51 step 4 figure 2
    short_channels = raw._short_channels()
    assert short_channels[0] is np.False_
    assert short_channels[2] is np.True_
    assert short_channels[4] is np.False_
    assert short_channels[6] is np.True_
    assert short_channels[8] is np.False_
    short_channels = raw._short_channels(threshold=3)
    assert short_channels[0] is np.False_
    assert short_channels[2] is np.False_
    short_channels = raw._short_channels(threshold=50)
    assert short_channels[0] is np.True_
    assert short_channels[2] is np.True_

    # Test trigger events
    assert raw.annotations[0]['description'] == '3.0'
    assert raw.annotations[1]['description'] == '2.0'
    assert raw.annotations[2]['description'] == '1.0'


@requires_testing_data
def test_nirx_standard():
    """Test standard operations."""
    _test_raw_reader(read_raw_nirx, fname=fname_nirx,
                     boundary_decimal=1)  # low fs


run_tests_if_main()
