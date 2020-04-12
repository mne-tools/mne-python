# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op
import pytest as pytest
import numpy as np
from numpy.testing import assert_allclose

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw
from mne.preprocessing.nirs import optical_density
from mne.utils import _validate_type
from mne.datasets import testing


fname_nirx = op.join(data_path(download=False),
                     'NIRx', 'nirx_15_2_recording_w_short')


@testing.requires_testing_data
def test_optical_density():
    """Test return type for optical density."""
    raw = read_raw_nirx(fname_nirx, preload=False)
    assert 'fnirs_raw' in raw
    assert 'fnirs_od' not in raw
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, 'raw')
    assert 'fnirs_raw' not in raw
    assert 'fnirs_od' in raw


@testing.requires_testing_data
def test_optical_density_zeromean():
    """Test that optical density can process zero mean data."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    raw._data[4] -= np.mean(raw._data[4])
    with pytest.warns(RuntimeWarning, match='Negative'):
        raw = optical_density(raw)
    assert 'fnirs_od' in raw


@testing.requires_testing_data
def test_optical_density_manual():
    """Test optical density on known values."""
    test_tol = 0.01
    raw = read_raw_nirx(fname_nirx, preload=True)
    # log(1) = 0
    raw._data[4] = np.ones((145))
    # log(0.5)/-1 = 0.69
    # log(1.5)/-1 = -0.40
    test_data = np.tile([0.5, 1.5], 73)[:145]
    raw._data[5] = test_data

    od = optical_density(raw)
    assert_allclose(od.get_data([4]), 0.)
    assert_allclose(od.get_data([5])[0, :2], [0.69, -0.4], atol=test_tol)
