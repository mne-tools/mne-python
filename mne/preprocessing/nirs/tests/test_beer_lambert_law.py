# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import pytest
import numpy as np

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, BaseRaw, read_raw_fif
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.utils import _validate_type
from mne.datasets import testing
from mne.externals.pymatreader import read_mat


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
def test_beer_lambert(fname, fmt, tmpdir):
    """Test converting NIRX files."""
    assert fmt in ('nirx', 'fif')
    raw = read_raw_nirx(fname)
    if fmt == 'fif':
        raw.save(tmpdir.join('test_raw.fif'))
        raw = read_raw_fif(tmpdir.join('test_raw.fif'))
    assert 'fnirs_cw_amplitude' in raw
    assert 'fnirs_od' not in raw
    raw = optical_density(raw)
    _validate_type(raw, BaseRaw, 'raw')
    assert 'fnirs_cw_amplitude' not in raw
    assert 'fnirs_od' in raw
    assert 'hbo' not in raw
    raw = beer_lambert_law(raw)
    _validate_type(raw, BaseRaw, 'raw')
    assert 'fnirs_cw_amplitude' not in raw
    assert 'fnirs_od' not in raw
    assert 'hbo' in raw
    assert 'hbr' in raw


@testing.requires_testing_data
def test_beer_lambert_unordered_errors():
    """NIRS data requires specific ordering and naming of channels."""
    raw = read_raw_nirx(fname_nirx_15_0)
    raw_od = optical_density(raw)
    raw_od.pick([0, 1, 2])
    with pytest.raises(ValueError, match='ordered'):
        beer_lambert_law(raw_od)

    # Test that an error is thrown if channel naming frequency doesn't match
    # what is stored in loc[9], which should hold the light frequency too.
    raw_od = optical_density(raw)
    raw_od.rename_channels({'S2_D2 760': 'S2_D2 770'})
    with pytest.raises(ValueError, match='frequency do not match'):
        beer_lambert_law(raw_od)

    # Test that an error is thrown if inconsistent frequencies used in data
    raw_od.info['chs'][2]['loc'][9] = 770.0
    with pytest.raises(ValueError, match='pairs with frequencies'):
        beer_lambert_law(raw_od)


@testing.requires_testing_data
def test_beer_lambert_v_matlab():
    """Compare MNE results to MATLAB toolbox."""
    raw = read_raw_nirx(fname_nirx_15_0)
    raw = optical_density(raw)
    raw = beer_lambert_law(raw, ppf=0.121)
    raw._data *= 1e6  # Scale to uM for comparison to MATLAB

    matlab_fname = op.join(data_path(download=False),
                           'NIRx', 'nirscout', 'validation',
                           'nirx_15_0_recording_bl.mat')
    matlab_data = read_mat(matlab_fname)

    for idx in range(raw.get_data().shape[0]):

        mean_error = np.mean(matlab_data['data'][:, idx] -
                             raw._data[idx])
        assert mean_error < 0.1
        matlab_name = ("S" + str(int(matlab_data['sources'][idx])) +
                       "_D" + str(int(matlab_data['detectors'][idx])) +
                       " " + matlab_data['type'][idx])
        assert raw.info['ch_names'][idx] == matlab_name
