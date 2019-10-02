# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os.path as op

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw
from mne.preprocessing import optical_density
from mne.utils import _validate_type
from mne.datasets import testing

from numpy import mean

fname_nirx = op.join(data_path(download=False),
                     'NIRx', 'nirx_15_2_recording_w_short')


@testing.requires_testing_data
def test_optical_density():
    """Test that the optical density conversion returns BaseRaw."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    od = optical_density(raw)
    _validate_type(od, BaseRaw, 'raw')


@testing.requires_testing_data
def test_optical_density_zeromean():
    """Test that optical density can process zero mean data."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    raw._data[4] -= mean(raw._data[4])
    od = optical_density(raw)
    _validate_type(od, BaseRaw, 'raw')
