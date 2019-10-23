# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import pytest

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw, read_raw_fif
from mne.preprocessing import optical_density, beer_lambert_law
from mne.utils import _validate_type
from mne.datasets import testing


fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirx_15_2_recording_w_short')


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2]))
@pytest.mark.parametrize('fmt', ('nirx', 'fif'))
def test_beer_lambert(fname, fmt, tmpdir):
    """Test converting NIRX files."""
    assert fmt in ('nirx', 'fif')
    raw = read_raw_nirx(fname)
    if fmt == 'fif':
        raw.save(tmpdir.join('test_raw.fif'))
        raw = read_raw_fif(tmpdir.join('test_raw.fif'))
    assert 'fnirs_raw' in raw
    assert 'fnirs_od' not in raw
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, 'raw')
    assert 'fnirs_raw' not in raw
    assert 'fnirs_od' in raw
    raw = beer_lambert_law(raw)
    _validate_type(raw, BaseRaw, 'raw')
    assert 'fnirs_raw' not in raw
    assert 'fnirs_od' not in raw
    assert 'hbo' in raw
    assert 'hbr' in raw
