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
    allowed_distance_error = 0.0002
    distances = raw._probe_distances()
    assert abs(distances[0] - 0.0304) < allowed_distance_error
    assert abs(distances[2] - 0.0078) < allowed_distance_error
    assert abs(distances[4] - 0.0310) < allowed_distance_error
    assert abs(distances[6] - 0.0086) < allowed_distance_error
    assert abs(distances[8] - 0.0416) < allowed_distance_error
    assert abs(distances[10] - 0.0072) < allowed_distance_error
    assert abs(distances[12] - 0.0389) < allowed_distance_error
    assert abs(distances[14] - 0.0075) < allowed_distance_error
    assert abs(distances[16] - 0.0558) < allowed_distance_error
    assert abs(distances[18] - 0.0562) < allowed_distance_error
    assert abs(distances[20] - 0.0561) < allowed_distance_error
    assert abs(distances[22] - 0.0565) < allowed_distance_error
    assert abs(distances[24] - 0.0077) < allowed_distance_error

    # Test which channels are short
    # These are the ones marked as red at
    # https://github.com/mne-tools/mne-testing-data/pull/51 step 4 figure 2
    short_channels = raw._short_channels()
    assert short_channels[0] is np.False_
    assert short_channels[2] is np.True_
    assert short_channels[4] is np.False_
    assert short_channels[6] is np.True_
    assert short_channels[8] is np.False_
    short_channels = raw._short_channels(threshold=0.003)
    assert short_channels[0] is np.False_
    assert short_channels[2] is np.False_
    short_channels = raw._short_channels(threshold=50)
    assert short_channels[0] is np.True_
    assert short_channels[2] is np.True_

    # Test trigger events
    assert raw.annotations[0]['description'] == '3.0'
    assert raw.annotations[1]['description'] == '2.0'
    assert raw.annotations[2]['description'] == '1.0'

    # Test location of detectors
    # The locations of detectors can be seen in the first
    # figure on this page...
    # https://github.com/mne-tools/mne-testing-data/pull/51
    # And have been manually copied below
    # These values were reported in mm, but according to this page...
    # https://mne.tools/stable/auto_tutorials/intro/plot_40_sensor_locations.html
    # 3d locations should be specified in meters, so that's what's tested below
    # Detector locations are stored in the third three loc values
    allowed_dist_error = 0.0002

    assert raw.info['ch_names'][0][3:5] == 'D1'
    assert abs(raw.info['chs'][0]['loc'][6] - -0.0841) < allowed_dist_error
    assert abs(raw.info['chs'][0]['loc'][7] - -0.0464) < allowed_dist_error
    assert abs(raw.info['chs'][0]['loc'][8] - -0.0129) < allowed_dist_error

    assert raw.info['ch_names'][8][3:5] == 'D2'
    assert abs(raw.info['chs'][8]['loc'][6] - 0.0207) < allowed_dist_error
    assert abs(raw.info['chs'][8]['loc'][7] - -0.1062) < allowed_dist_error
    assert abs(raw.info['chs'][8]['loc'][8] - 0.0484) < allowed_dist_error

    assert raw.info['ch_names'][4][3:5] == 'D3'
    assert abs(raw.info['chs'][4]['loc'][6] - 0.0846) < allowed_dist_error
    assert abs(raw.info['chs'][4]['loc'][7] - -0.0142) < allowed_dist_error
    assert abs(raw.info['chs'][4]['loc'][8] - -0.0156) < allowed_dist_error

    assert raw.info['ch_names'][12][3:5] == 'D4'
    assert abs(raw.info['chs'][12]['loc'][6] - -0.0196) < allowed_dist_error
    assert abs(raw.info['chs'][12]['loc'][7] - 0.0821) < allowed_dist_error
    assert abs(raw.info['chs'][12]['loc'][8] - 0.0275) < allowed_dist_error

    assert raw.info['ch_names'][16][3:5] == 'D5'
    assert abs(raw.info['chs'][16]['loc'][6] - -0.0360) < allowed_dist_error
    assert abs(raw.info['chs'][16]['loc'][7] - 0.0276) < allowed_dist_error
    assert abs(raw.info['chs'][16]['loc'][8] - 0.0778) < allowed_dist_error

    assert raw.info['ch_names'][19][3:5] == 'D6'
    assert abs(raw.info['chs'][19]['loc'][6] - 0.0352) < allowed_dist_error
    assert abs(raw.info['chs'][19]['loc'][7] - 0.0283) < allowed_dist_error
    assert abs(raw.info['chs'][19]['loc'][8] - 0.0780) < allowed_dist_error

    assert raw.info['ch_names'][21][3:5] == 'D7'
    assert abs(raw.info['chs'][21]['loc'][6] - 0.0388) < allowed_dist_error
    assert abs(raw.info['chs'][21]['loc'][7] - -0.0477) < allowed_dist_error
    assert abs(raw.info['chs'][21]['loc'][8] - 0.0932) < allowed_dist_error


@requires_testing_data
def test_nirx_standard():
    """Test standard operations."""
    _test_raw_reader(read_raw_nirx, fname=fname_nirx,
                     boundary_decimal=1)  # low fs


run_tests_if_main()
