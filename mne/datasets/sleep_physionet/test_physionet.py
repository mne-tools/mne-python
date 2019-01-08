import pytest
import numpy as np
import os.path as op

from numpy.testing import assert_array_equal
from .age.sleep_physionet_age import _update_sleep_records
from .age.sleep_physionet_age import fetch_data
from ...utils import _TempDir, run_tests_if_main, requires_good_network

@requires_good_network
def test_run_update():
    """Test Sleep Physionet URL handling."""
    _update_sleep_records()


def _keep_basename_only(path_structure):
    return np.vectorize(op.basename)(np.array(path_structure))

@requires_good_network
def test_sleep_physionet_age():
    """Test Sleep Physionet URL handling."""

    data_dir = _TempDir()
    paths = fetch_data(subjects=[0], record=[1], path=data_dir, update_path=False)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf']])

    paths = fetch_data(subjects=[0, 1], record=[1], path=data_dir, update_path=False)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4011E0-PSG.edf', 'SC4011EH-Hypnogram.edf']])

    paths = fetch_data(subjects=[0], record=[1, 2], path=data_dir,
                       update_path=False)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4002E0-PSG.edf', 'SC4002EC-Hypnogram.edf']])



run_tests_if_main()
