"""Data Equivalence Tests"""
from __future__ import print_function

# Authors: alexandre barachant
#
# License: BSD (3-clause)

import os.path as op
import inspect
import warnings

from numpy.testing import assert_array_almost_equal
import numpy as np

from mne.io import read_raw_edf
from mne.utils import run_tests_if_main
from mne import pick_types

warnings.simplefilter('always')

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
gdf_path = op.join(data_dir, 'test.gdf')
gdf_biosig_path = op.join(data_dir, 'test_gdf_biosig.npy')


def test_gdf_data():
    """Test reading raw gdf files
    """
    raw_py = read_raw_edf(gdf_path, eog=None,
                          misc=None, preload=True, stim_channel=None)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    # this .npy was generated using the official biosig python package
    raw_biosig = np.load(gdf_biosig_path)
    raw_biosig = raw_biosig * 1e-6  # data are stored in microvolts
    data_biosig = raw_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data_py, data_biosig, 8)

run_tests_if_main()
