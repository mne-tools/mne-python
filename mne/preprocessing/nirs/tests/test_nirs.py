# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import pytest

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import optical_density, beer_lambert_law,\
    _fnirs_check_bads, _fnirs_spread_bads

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
def test_fnirs_check_bads(fname):
    """Test checking of bad markings."""
    # No bad channels, so these should all pass
    raw = read_raw_nirx(fname)
    _fnirs_check_bads(raw)
    raw = optical_density(raw)
    _fnirs_check_bads(raw)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw)

    # Mark pairs of bad channels, so these should all pass
    raw = read_raw_nirx(fname)
    raw.info['bads'] = raw.ch_names[0:2]
    _fnirs_check_bads(raw)
    raw = optical_density(raw)
    _fnirs_check_bads(raw)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw)

    # Mark single channel as bad, so these should all fail
    raw = read_raw_nirx(fname)
    raw.info['bads'] = raw.ch_names[0:1]
    pytest.raises(RuntimeError, _fnirs_check_bads, raw)
    raw = optical_density(raw)
    pytest.raises(RuntimeError, _fnirs_check_bads, raw)
    raw = beer_lambert_law(raw)
    pytest.raises(RuntimeError, _fnirs_check_bads, raw)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
def test_fnirs_spread_bads(fname):
    """Test checking of bad markings."""
    # Test spreading upwards in frequency and on raw data
    raw = read_raw_nirx(fname)
    raw.info['bads'] = ['S1_D1 760']
    raw = _fnirs_spread_bads(raw)
    assert raw.info['bads'] == ['S1_D1 760', 'S1_D1 850']

    # Test spreading downwards in frequency and on od data
    raw = optical_density(raw)
    raw.info['bads'] = raw.ch_names[5:6]
    raw = _fnirs_spread_bads(raw)
    assert raw.info['bads'] == raw.ch_names[4:6]

    # Test spreading multiple bads and on chroma data
    raw = beer_lambert_law(raw)
    raw.info['bads'] = [raw.ch_names[x] for x in [1, 8]]
    print(raw.info['bads'])
    raw = _fnirs_spread_bads(raw)
    print(raw.info['bads'])
    assert raw.info['bads'] == [raw.ch_names[x] for x in [0, 1, 8, 9]]
