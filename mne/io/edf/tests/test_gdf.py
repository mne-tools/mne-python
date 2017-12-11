"""Data Equivalence Tests"""
from __future__ import print_function

# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from nose.tools import assert_true
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
import numpy as np
import scipy.io as sio

from mne.datasets import testing
from mne.io import read_raw_edf
from mne.utils import run_tests_if_main
from mne import pick_types, find_events

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
gdf1_path = op.join(data_path, 'GDF', 'test_gdf_1.25')
gdf2_path = op.join(data_path, 'GDF', 'test_gdf_2.20')


@testing.requires_testing_data
def test_gdf_data():
    """Test reading raw GDF 1.x files."""
    with warnings.catch_warnings(record=True):  # interpolate / overlap events
        raw = read_raw_edf(gdf1_path + '.gdf', eog=None,
                           misc=None, preload=True, stim_channel='auto')
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # this .npy was generated using the official biosig python package
    raw_biosig = np.load(gdf1_path + '_biosig.npy')
    raw_biosig = raw_biosig * 1e-6  # data are stored in microvolts
    data_biosig = raw_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Test for stim channel
    events = find_events(raw, shortest_event=1)
    # The events are overlapping.
    assert_array_equal(events[:, 0], raw._raw_extras[0]['events'][1][::2])

    # Test events are encoded to stim channel.
    events = find_events(raw)
    evs = raw.find_edf_events()
    assert_true(all([event in evs[1] for event in events[:, 0]]))


@testing.requires_testing_data
def test_gdf2_data():
    """Test reading raw GDF 2.x files."""
    raw = read_raw_edf(gdf2_path + '.gdf', eog=None, misc=None, preload=True,
                       stim_channel='STATUS')

    nchan = raw.info['nchan']
    ch_names = raw.ch_names  # Renamed STATUS -> STI 014.
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # This .mat was generated using the official biosig matlab package
    mat = sio.loadmat(gdf2_path + '_biosig.mat')
    data_biosig = mat['dat'] * 1e-6  # data are stored in microvolts
    data_biosig = data_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Find events
    events = find_events(raw, verbose=1)
    events[:, 2] >>= 8  # last 8 bits are system events in biosemi files
    assert_equal(events.shape[0], 2)  # 2 events in file
    assert_array_equal(events[:, 2], [20, 28])

    with warnings.catch_warnings(record=True) as w:
        # header contains no events
        raw = read_raw_edf(gdf2_path + '.gdf', stim_channel='auto')
        assert_equal(len(w), 1)
        assert_true(str(w[0].message).startswith('No events found.'))
    assert_equal(nchan, raw.info['nchan'])  # stim channel not constructed
    assert_array_equal(ch_names[1:], raw.ch_names[1:])


run_tests_if_main()
