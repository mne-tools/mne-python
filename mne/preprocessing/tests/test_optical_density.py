# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os.path as op

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw
from mne.preprocessing import optical_density
from mne.utils import _validate_type

fname_nirx = op.join(data_path(download=False),
                     'NIRx', 'nirx_15_2_recording_w_short')


def test_optical_density():
    """Test fix stim artifact."""
    raw = read_raw_nirx(fname_nirx, preload=True)
    od = optical_density(raw, verbose=None)
    _validate_type(od, BaseRaw, 'raw')
    print(od)
