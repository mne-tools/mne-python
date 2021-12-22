# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD-3-Clause

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_allclose

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import optical_density, tddr
from mne.datasets import testing


fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2]))
def test_temporal_derivative_distribution_repair(fname, tmp_path):
    """Test running artifact rejection."""
    raw = read_raw_nirx(fname)
    raw = optical_density(raw)

    # Add a baseline shift artifact about half way through data
    max_shift = np.max(np.diff(raw._data[0]))
    shift_amp = 5 * max_shift
    raw._data[0, 0:30] = raw._data[0, 0:30] - (shift_amp)
    # make one channel zero std
    raw._data[1] = 0.
    raw._data[2] = 1.
    assert np.max(np.diff(raw._data[0])) > shift_amp
    # Ensure that applying the algorithm reduces the step change
    raw = tddr(raw)
    assert np.max(np.diff(raw._data[0])) < shift_amp
    assert_allclose(raw._data[1], 0.)  # unchanged
    assert_allclose(raw._data[2], 1.)  # unchanged
