# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import optical_density, scalp_coupling_index,\
    beer_lambert_law
from mne.datasets import testing

fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout',
                                'nirx_15_2_recording_w_short')


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
@pytest.mark.parametrize('fmt', ('nirx', 'fif'))
def test_scalp_coupling_index(fname, fmt, tmp_path):
    """Test converting NIRX files."""
    assert fmt in ('nirx', 'fif')
    raw = read_raw_nirx(fname)
    with pytest.raises(RuntimeError, match='Scalp'):
        scalp_coupling_index(raw)

    raw = optical_density(raw)
    sci = scalp_coupling_index(raw)

    # All values should be between -1 and +1
    assert_array_less(sci, 1.0)
    assert_array_less(sci * -1.0, 1.0)

    # Fill in some data with known correlation values
    rng = np.random.RandomState(0)
    new_data = rng.rand(raw._data[0].shape[0])
    # Set first two channels to perfect correlation
    raw._data[0] = new_data
    raw._data[1] = new_data
    # Set next two channels to perfect correlation
    raw._data[2] = new_data
    raw._data[3] = new_data * 0.3  # check scale invariance
    # Set next two channels to anti correlation
    raw._data[4] = new_data
    raw._data[5] = new_data * -1.0
    # Set next two channels to be uncorrelated
    raw._data[6] = new_data
    raw._data[7] = rng.rand(raw._data[0].shape[0])
    # Set next channel to have zero std
    raw._data[8] = 0.
    raw._data[9] = 1.
    raw._data[10] = 2.
    raw._data[11] = 3.
    # Check values
    sci = scalp_coupling_index(raw)
    assert_allclose(sci[0:6], [1, 1, 1, 1, -1, -1], atol=0.01)
    assert np.abs(sci[6]) < 0.5
    assert np.abs(sci[7]) < 0.5
    assert_allclose(sci[8:12], 0, atol=1e-10)

    # Ensure function errors if wrong type is passed in
    raw = beer_lambert_law(raw, ppf=6)
    with pytest.raises(RuntimeError, match='Scalp'):
        scalp_coupling_index(raw)
