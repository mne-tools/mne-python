# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

from functools import partial
import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from mne import create_info
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx, RawArray
from mne.preprocessing.nirs import (optical_density, beer_lambert_law as _bll,
                                    _fnirs_check_bads, _fnirs_spread_bads,
                                    _check_channels_ordered,
                                    _channel_frequencies, _channel_chromophore)
from mne.io.pick import _picks_to_idx

from mne.datasets import testing
from mne.io.constants import FIFF

beer_lambert_law = partial(_bll, ppf=6)  # temporary deprecation wrapper

fname_nirx_15_0 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_0_recording')
fname_nirx_15_2 = op.join(data_path(download=False),
                          'NIRx', 'nirscout', 'nirx_15_2_recording')
fname_nirx_15_2_short = op.join(data_path(download=False),
                                'NIRx', 'nirscout',
                                'nirx_15_2_recording_w_short')


@testing.requires_testing_data
def test_fnirs_picks():
    """Test picking of fnirs types after different conversions."""
    raw = read_raw_nirx(fname_nirx_15_0)
    picks = _picks_to_idx(raw.info, 'fnirs_cw_amplitude')
    assert len(picks) == len(raw.ch_names)
    raw_subset = raw.copy().pick(picks='fnirs_cw_amplitude')
    for ch in raw_subset.info["chs"]:
        assert ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE

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
    raw_subset = raw.copy().pick(picks='fnirs_od')
    for ch in raw_subset.info["chs"]:
        assert ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_OD

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
    raw_subset = raw.copy().pick(picks='hbo')
    for ch in raw_subset.info["chs"]:
        assert ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBO

    picks = _picks_to_idx(raw.info, ['hbr'])
    assert len(picks) == len(raw.ch_names) / 2
    raw_subset = raw.copy().pick(picks=['hbr'])
    for ch in raw_subset.info["chs"]:
        assert ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBR

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
    _fnirs_check_bads(raw.info)
    raw = optical_density(raw)
    _fnirs_check_bads(raw.info)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw.info)

    # Mark pairs of bad channels, so these should all pass
    raw = read_raw_nirx(fname)
    raw.info['bads'] = raw.ch_names[0:2]
    _fnirs_check_bads(raw.info)
    raw = optical_density(raw)
    _fnirs_check_bads(raw.info)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw.info)

    # Mark single channel as bad, so these should all fail
    raw = read_raw_nirx(fname)
    raw.info['bads'] = raw.ch_names[0:1]
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)
    with pytest.raises(RuntimeError, match='bad labelling'):
        raw = optical_density(raw)
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)
    with pytest.raises(RuntimeError, match='bad labelling'):
        raw = beer_lambert_law(raw)
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
def test_fnirs_spread_bads(fname):
    """Test checking of bad markings."""
    # Test spreading upwards in frequency and on raw data
    raw = read_raw_nirx(fname)
    raw.info['bads'] = ['S1_D1 760']
    info = _fnirs_spread_bads(raw.info)
    assert info['bads'] == ['S1_D1 760', 'S1_D1 850']

    # Test spreading downwards in frequency and on od data
    raw = optical_density(raw)
    raw.info['bads'] = raw.ch_names[5:6]
    info = _fnirs_spread_bads(raw.info)
    assert info['bads'] == raw.ch_names[4:6]

    # Test spreading multiple bads and on chroma data
    raw = beer_lambert_law(raw)
    raw.info['bads'] = [raw.ch_names[x] for x in [1, 8]]
    info = _fnirs_spread_bads(raw.info)
    assert info['bads'] == [info.ch_names[x] for x in [0, 1, 8, 9]]


@testing.requires_testing_data
@pytest.mark.parametrize('fname', ([fname_nirx_15_2_short, fname_nirx_15_2,
                                    fname_nirx_15_0]))
def test_fnirs_channel_naming_and_order_readers(fname):
    """Ensure fNIRS channel checking on standard readers."""
    # fNIRS data requires specific channel naming and ordering.

    # All standard readers should pass tests
    raw = read_raw_nirx(fname)
    freqs = np.unique(_channel_frequencies(raw.info))
    assert_array_equal(freqs, [760, 850])
    chroma = np.unique(_channel_chromophore(raw.info))
    assert len(chroma) == 0

    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)  # as all fNIRS only data

    # Check that dropped channels are detected
    # For each source detector pair there must be two channels,
    # removing one should throw an error.
    raw_dropped = raw.copy().drop_channels(raw.ch_names[4])
    with pytest.raises(ValueError, match='not ordered correctly'):
        _check_channels_ordered(raw_dropped.info, freqs)

    # The ordering must match the passed in argument
    raw_names_reversed = raw.copy().ch_names
    raw_names_reversed.reverse()
    raw_reversed = raw.copy().pick_channels(raw_names_reversed, ordered=True)
    with pytest.raises(ValueError, match='not ordered .* frequencies'):
        _check_channels_ordered(raw_reversed.info, freqs)
    # So if we flip the second argument it should pass again
    _check_channels_ordered(raw_reversed.info, [850, 760])

    # Check on OD data
    raw = optical_density(raw)
    freqs = np.unique(_channel_frequencies(raw.info))
    assert_array_equal(freqs, [760, 850])
    chroma = np.unique(_channel_chromophore(raw.info))
    assert len(chroma) == 0
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)  # as all fNIRS only data

    # Check on haemoglobin data
    raw = beer_lambert_law(raw)
    freqs = np.unique(_channel_frequencies(raw.info))
    assert len(freqs) == 0
    assert len(_channel_chromophore(raw.info)) == len(raw.ch_names)
    chroma = np.unique(_channel_chromophore(raw.info))
    assert_array_equal(chroma, ["hbo", "hbr"])
    picks = _check_channels_ordered(raw.info, chroma)
    assert len(picks) == len(raw.ch_names)
    with pytest.raises(ValueError, match='not ordered .* chromophore'):
        _check_channels_ordered(raw.info, ["hbx", "hbr"])


def test_fnirs_channel_naming_and_order_custom_raw():
    """Ensure fNIRS channel checking on manually created data."""
    data = np.random.normal(size=(6, 10))

    # Start with a correctly named raw intensity dataset
    # These are the steps required to build an fNIRS Raw object from scratch
    ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850',
                'S3_D1 760', 'S3_D1 850']
    ch_types = np.repeat("fnirs_cw_amplitude", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.tile([760, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f

    freqs = np.unique(_channel_frequencies(raw.info))
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)
    assert len(picks) == 6

    # Different systems use different frequencies, so ensure that works
    ch_names = ['S1_D1 920', 'S1_D1 850', 'S2_D1 920', 'S2_D1 850',
                'S3_D1 920', 'S3_D1 850']
    ch_types = np.repeat("fnirs_cw_amplitude", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.tile([920, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f

    picks = _check_channels_ordered(raw.info, [920, 850])
    assert len(picks) == len(raw.ch_names)
    assert len(picks) == 6

    # Catch expected errors

    # The frequencies named in the channel names must match the info loc field
    ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850',
                'S3_D1 760', 'S3_D1 850']
    ch_types = np.repeat("fnirs_cw_amplitude", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.tile([920, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f
    with pytest.raises(ValueError, match='not ordered'):
        _check_channels_ordered(raw.info, [920, 850])

    # Catch if someone doesn't set the info field
    ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850',
                'S3_D1 760', 'S3_D1 850']
    ch_types = np.repeat("fnirs_cw_amplitude", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    with pytest.raises(ValueError, match='missing wavelength information'):
        _check_channels_ordered(raw.info, [920, 850])

    # I have seen data encoded not in alternating frequency, but blocked.
    ch_names = ['S1_D1 760', 'S2_D1 760', 'S3_D1 760',
                'S1_D1 850', 'S2_D1 850', 'S3_D1 850']
    ch_types = np.repeat("fnirs_cw_amplitude", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.repeat([760, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f
    with pytest.raises(ValueError, match='channels not ordered correctly'):
        _check_channels_ordered(raw.info, [760, 850])
    # and this is how you would fix the ordering, then it should pass
    raw.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw.info, [760, 850])


def test_fnirs_channel_naming_and_order_custom_optical_density():
    """Ensure fNIRS channel checking on manually created data."""
    data = np.random.normal(size=(6, 10))

    # Start with a correctly named raw intensity dataset
    # These are the steps required to build an fNIRS Raw object from scratch
    ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850',
                'S3_D1 760', 'S3_D1 850']
    ch_types = np.repeat("fnirs_od", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.tile([760, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f

    freqs = np.unique(_channel_frequencies(raw.info))
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)
    assert len(picks) == 6

    # Check block naming for optical density
    ch_names = ['S1_D1 760', 'S2_D1 760', 'S3_D1 760',
                'S1_D1 850', 'S2_D1 850', 'S3_D1 850']
    ch_types = np.repeat("fnirs_od", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    freqs = np.repeat([760, 850], 3)
    for idx, f in enumerate(freqs):
        raw.info["chs"][idx]["loc"][9] = f
    with pytest.raises(ValueError, match='channels not ordered correctly'):
        _check_channels_ordered(raw.info, [760, 850])
    # and this is how you would fix the ordering, then it should pass
    raw.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw.info, [760, 850])

    # Check that if you mix types you get an error
    ch_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr',
                'S3_D1 hbo', 'S3_D1 hbr']
    ch_types = np.tile(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw2 = RawArray(data, info, verbose=True)
    raw.add_channels([raw2])
    with pytest.raises(ValueError, match='does not support a combination'):
        _check_channels_ordered(raw.info, [760, 850])


def test_fnirs_channel_naming_and_order_custom_chroma():
    """Ensure fNIRS channel checking on manually created data."""
    data = np.random.RandomState(0).randn(6, 10)

    # Start with a correctly named raw intensity dataset
    # These are the steps required to build an fNIRS Raw object from scratch
    ch_names = ['S1_D1 hbo', 'S1_D1 hbr', 'S2_D1 hbo', 'S2_D1 hbr',
                'S3_D1 hbo', 'S3_D1 hbr']
    ch_types = np.tile(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)

    chroma = np.unique(_channel_chromophore(raw.info))
    picks = _check_channels_ordered(raw.info, chroma)
    assert len(picks) == len(raw.ch_names)
    assert len(picks) == 6

    # Test block creation fails
    ch_names = ['S1_D1 hbo', 'S2_D1 hbo', 'S3_D1 hbo',
                'S1_D1 hbr', 'S2_D1 hbr', 'S3_D1 hbr']
    ch_types = np.repeat(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    with pytest.raises(ValueError, match='not ordered .* chromophore'):
        _check_channels_ordered(raw.info, ["hbo", "hbr"])
    # Reordering should fix
    raw.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw.info, ["hbo", "hbr"])
    # Wrong names should fail
    with pytest.raises(ValueError, match='not ordered .* chromophore'):
        _check_channels_ordered(raw.info, ["hbb", "hbr"])

    # Test weird naming
    ch_names = ['S1_D1 hbb', 'S1_D1 hbr', 'S2_D1 hbb', 'S2_D1 hbr',
                'S3_D1 hbb', 'S3_D1 hbr']
    ch_types = np.tile(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    with pytest.raises(ValueError, match='naming conventions'):
        _check_channels_ordered(raw.info, ["hbb", "hbr"])

    # Check more weird naming
    ch_names = ['S1_DX hbo', 'S1_DX hbr', 'S2_D1 hbo', 'S2_D1 hbr',
                'S3_D1 hbo', 'S3_D1 hbr']
    ch_types = np.tile(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    raw = RawArray(data, info, verbose=True)
    with pytest.raises(ValueError, match='can not be parsed'):
        _check_channels_ordered(raw.info, ["hbo", "hbr"])
