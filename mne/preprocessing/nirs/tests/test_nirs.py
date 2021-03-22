# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    _fnirs_check_bads, _fnirs_spread_bads,
                                    _check_channels_ordered,
                                    _channel_frequencies, _channel_chromophore)
from mne.io.pick import _picks_to_idx

from mne.datasets import testing

fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout',
                                'nirx_15_2_recording_w_short')


def test_fnirs_picks():
    """Test picking of fnirs types after different conversions"""
    raw = read_raw_nirx(fname_nirx_15_0)
    picks = _picks_to_idx(raw.info, 'fnirs_cw_amplitude')
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['fnirs_cw_amplitude', 'fnirs_od'])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['fnirs_cw_amplitude', 'fnirs_od', 'hbr'])
    assert len(picks) == len(raw.ch_names)
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_od')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'hbo')
    pytest.raises(ValueError, _picks_to_idx, raw.info, ['hbr'])
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_fd_phase')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'junk')

    raw = optical_density(raw)
    picks = _picks_to_idx(raw.info, 'fnirs_od')
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['fnirs_cw_amplitude', 'fnirs_od'])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['fnirs_cw_amplitude', 'fnirs_od', 'hbr'])
    assert len(picks) == len(raw.ch_names)
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_cw_amplitude')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'hbo')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'hbr')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_fd_phase')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'junk')

    raw = beer_lambert_law(raw)
    picks = _picks_to_idx(raw.info, 'hbo')
    assert len(picks) == len(raw.ch_names) / 2
    picks = _picks_to_idx(raw.info, ['hbr'])
    assert len(picks) == len(raw.ch_names) / 2
    picks = _picks_to_idx(raw.info, ['hbo', 'hbr'])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['hbo', 'fnirs_od', 'hbr'])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ['hbo', 'fnirs_od'])
    assert len(picks) == len(raw.ch_names) / 2
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_cw_amplitude')
    pytest.raises(ValueError, _picks_to_idx, raw.info, ['fnirs_od'])
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'junk')
    pytest.raises(ValueError, _picks_to_idx, raw.info, 'fnirs_fd_phase')


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
    raw = _fnirs_spread_bads(raw)
    assert raw.info['bads'] == [raw.ch_names[x] for x in [0, 1, 8, 9]]


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
def test_fnirs_channel_naming_and_order_readers(fname):
    """Ensure fNIRS channel checking on standard readers"""
    # fNIRS data requires specific channel naming and ordering.

    # All standard readers should pass tests
    raw = read_raw_nirx(fname)
    freqs = np.unique(_channel_frequencies(raw))
    assert_array_equal(freqs, [760, 850])

    picks = _check_channels_ordered(raw, freqs)
    assert len(picks) == len(raw.ch_names)  # as all fNIRS only data

    # Check that dropped channels are detected
    # For each source detector pair there must be two channels,
    # removing one should throw an error.
    raw_dropped = raw.copy().drop_channels(raw.ch_names[4])
    with pytest.raises(ValueError, match='not ordered correctly'):
        _check_channels_ordered(raw_dropped, freqs)

    # The ordering must match the passed in argument
    raw_names_reversed = raw.copy().ch_names
    raw_names_reversed.reverse()
    raw_reversed = raw.copy().pick_channels(raw_names_reversed, ordered=True)
    with pytest.raises(ValueError, match='not ordered correctly'):
        _check_channels_ordered(raw_reversed, freqs)
    # So if we flip the second argument it should pass again
    _check_channels_ordered(raw_reversed, [850, 760])

    # Check on OD data
    raw = optical_density(raw)
    freqs = np.unique(_channel_frequencies(raw))
    assert_array_equal(freqs, [760, 850])
    picks = _check_channels_ordered(raw, freqs)
    assert len(picks) == len(raw.ch_names)  # as all fNIRS only data

    # Check on haemoglobin data
    raw = beer_lambert_law(raw)
    freqs = np.unique(_channel_chromophore(raw))
    assert_array_equal(freqs, ["hbo", "hbr"])
