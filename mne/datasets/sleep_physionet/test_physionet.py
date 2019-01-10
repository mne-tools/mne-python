import pytest
import numpy as np
import os.path as op

from numpy.testing import assert_array_equal
from ...utils import _TempDir, run_tests_if_main, requires_good_network
from .age.sleep_physionet_age import _update_sleep_records as age_records
from .age.sleep_physionet_age import fetch_data as fetch_age_data
from .temazepam.sleep_physionet_temazepam import _update_sleep_records \
    as temazepam_records
from .temazepam.sleep_physionet_temazepam import fetch_data \
    as fetch_temazepam_data


def _keep_basename_only(path_structure):
    return np.vectorize(op.basename)(np.array(path_structure))


@requires_good_network
def test_run_update_age_records():
    """Test Sleep Physionet URL handling."""
    age_records()


@requires_good_network
def test_sleep_physionet_age():
    """Test Sleep Physionet URL handling."""
    params = {'path': _TempDir(), 'update_path': False}

    paths = fetch_age_data(subjects=[0], record=[1], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf']])

    paths = fetch_age_data(subjects=[0, 1], record=[1], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4011E0-PSG.edf', 'SC4011EH-Hypnogram.edf']])

    paths = fetch_age_data(subjects=[0], record=[1, 2], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4002E0-PSG.edf', 'SC4002EC-Hypnogram.edf']])


@requires_good_network
def test_run_update_temazepam_records():
    """Test Sleep Physionet URL handling."""
    temazepam_records()


@requires_good_network
def test_sleep_physionet_temazepam():
    """Test Sleep Physionet URL handling."""
    params = {'path': _TempDir(), 'update_path': False}

    paths = fetch_temazepam_data(subjects=[1], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['ST7011J0-PSG.edf', 'ST7011JP-Hypnogram.edf']])

    with pytest.raises(RuntimeError, match='Unknown subjects: 0, 3'):
        paths = fetch_temazepam_data(subjects=list(range(4)), **params)

run_tests_if_main()
