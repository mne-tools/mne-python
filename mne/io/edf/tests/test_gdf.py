"""Data Equivalence Tests"""
from __future__ import print_function

# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from numpy.testing import assert_array_almost_equal, assert_array_equal
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
    raw = read_raw_edf(gdf1_path + '.gdf', eog=None,
                       misc=None, preload=True, stim_channel=None)
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # this .npy was generated using the official biosig python package
    raw_biosig = np.load(gdf1_path + '_biosig.npy')
    raw_biosig = raw_biosig * 1e-6  # data are stored in microvolts
    data_biosig = raw_biosig[picks]

    # Assert data are almost equal
    print(data.shape)
    print(data_biosig.shape)
    assert_array_almost_equal(data, data_biosig, 8)


@testing.requires_testing_data
def test_gdf2_data():
    """Test reading raw GDF 2.x files."""
    raw = read_raw_edf(gdf2_path + '.gdf', eog=None, misc=None, preload=True,
                       stim_channel='STATUS', biosemi=True)
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # This .mat was generated using the official biosig matlab package
    mat = sio.loadmat(gdf2_path + '_biosig.mat')
    data_biosig = mat['dat'] * 1e-6  # data are stored in microvolts
    data_biosig = data_biosig[picks]

    # Assert data are almost equal
    print(data.shape)
    print(data_biosig.shape)
    assert_array_almost_equal(data, data_biosig, 8)

    # Find events
    events = find_events(raw, verbose=1)
    assert(events.shape[0] == 2)  # 2 events in file
    assert_array_equal(events[:, 2], [20, 28])


run_tests_if_main()
