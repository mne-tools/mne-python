# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause


import pytest

import numpy as np

from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_raises)

from mne import create_info
from mne.channels import make_standard_montage, compute_native_head_t
from mne.channels.montage import get_builtin_montages, HEAD_SIZE_DEFAULT
from mne.io import RawArray
from mne.io._digitization import _get_dig_eeg, _get_fid_coords
from mne.io.constants import FIFF
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.transforms import _get_trans, _angle_between_quats, rot_to_quat


@pytest.mark.parametrize('kind', get_builtin_montages())
def test_standard_montages_have_fids(kind):
    """Test standard montage are all in unknown coord (have fids)."""
    montage = make_standard_montage(kind)
    fids, coord_frame = _get_fid_coords(montage.dig)
    for k, v in fids.items():
        assert v is not None, k
    for d in montage.dig:
        if kind == 'artinis-octamon' or kind == 'artinis-brite23':
            assert d['coord_frame'] == FIFF.FIFFV_COORD_MRI
        else:
            assert d['coord_frame'] == FIFF.FIFFV_COORD_UNKNOWN


def test_standard_montage_errors():
    """Test error handling for wrong keys."""
    _msg = "Invalid value for the 'kind' parameter..*but got.*not-here"
    with pytest.raises(ValueError, match=_msg):
        _ = make_standard_montage('not-here')


@pytest.mark.parametrize('head_size', (HEAD_SIZE_DEFAULT, 0.05))
@pytest.mark.parametrize('kind, tol', [
    ['EGI_256', 1e-5],
    ['easycap-M1', 1e-8],
    ['easycap-M10', 1e-8],
    ['biosemi128', 1e-8],
    ['biosemi16', 1e-8],
    ['biosemi160', 1e-8],
    ['biosemi256', 1e-8],
    ['biosemi32', 1e-8],
    ['biosemi64', 1e-8],
])
def test_standard_montages_on_sphere(kind, tol, head_size):
    """Test some standard montage are on sphere."""
    kwargs = dict()
    if head_size != HEAD_SIZE_DEFAULT:
        kwargs['head_size'] = head_size
    montage = make_standard_montage(kind, **kwargs)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), head_size),
        atol=tol,
    )


def test_standard_superset():
    """Test some properties that should hold for superset montages."""
    # new montages, tweaked to end up at the same size as the others
    m_1005 = make_standard_montage('standard_1005', 0.0970)
    m_1020 = make_standard_montage('standard_1020', 0.0991)
    assert len(set(m_1005.ch_names) - set(m_1020.ch_names)) > 0
    # XXX weird that this is not a proper superset...
    assert set(m_1020.ch_names) - set(m_1005.ch_names) == {'O10', 'O9'}
    c_1005 = m_1005._get_ch_pos()
    for key, value in m_1020._get_ch_pos().items():
        if key not in ('O10', 'O9'):
            assert_allclose(c_1005[key], value, atol=1e-4, err_msg=key)


def _simulate_artinis_octamon():
    """Simulate artinis OctaMon channel data from numpy data.

    This is to test data that is imported with missing or incorrect montage
    info. This data can then be used to test the set_montage function.
    """
    np.random.seed(42)
    data = np.absolute(np.random.normal(size=(16, 100)))
    ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850',
                'S3_D1 760', 'S3_D1 850', 'S4_D1 760', 'S4_D1 850',
                'S5_D2 760', 'S5_D2 850', 'S6_D2 760', 'S6_D2 850',
                'S7_D2 760', 'S7_D2 850', 'S8_D2 760', 'S8_D2 850']
    ch_types = ['fnirs_cw_amplitude' for _ in ch_names]
    sfreq = 10.  # Hz
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    for i, ch_name in enumerate(ch_names):
        info['chs'][i]['loc'][9] = int(ch_name.split(' ')[1])
    raw = RawArray(data, info)

    return raw


def _simulate_artinis_brite23():
    """Simulate artinis Brite23 channel data from numpy data.

    This is to test data that is imported with missing or incorrect montage
    info. This data can then be used to test the set_montage function.
    """
    np.random.seed(0)
    data = np.random.normal(size=(46, 100))
    sd_names = ['S1_D1', 'S2_D1', 'S3_D1', 'S4_D1', 'S3_D2', 'S4_D2', 'S5_D2',
                'S4_D3', 'S5_D3', 'S6_D3', 'S5_D4', 'S6_D4', 'S7_D4', 'S6_D5',
                'S7_D5', 'S8_D5', 'S7_D6', 'S8_D6', 'S9_D6', 'S8_D7', 'S9_D7',
                'S10_D7', 'S11_D7']
    ch_names = []
    ch_types = []
    for name in sd_names:
        ch_names.append(name + ' hbo')
        ch_types.append('hbo')
        ch_names.append(name + ' hbr')
        ch_types.append('hbr')
    sfreq = 10.  # Hz
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data, info)

    return raw


@pytest.mark.parametrize('kind', ('octamon', 'brite23'))
def test_set_montage_artinis_fsaverage(kind):
    """Test that artinis montages match fsaverage's head<->MRI transform."""
    # Compare OctaMon and Brite23 to fsaverage
    trans_fs, _ = _get_trans('fsaverage')
    montage = make_standard_montage(f'artinis-{kind}')
    trans = compute_native_head_t(montage)
    assert trans['to'] == trans_fs['to']
    assert trans['from'] == trans_fs['from']
    translation = 1000 * np.linalg.norm(trans['trans'][:3, 3] -
                                        trans_fs['trans'][:3, 3])
    # TODO: This is actually quite big...
    assert 15 < translation < 18  # mm
    rotation = np.rad2deg(
        _angle_between_quats(rot_to_quat(trans['trans'][:3, :3]),
                             rot_to_quat(trans_fs['trans'][:3, :3])))
    assert 3 < rotation < 7  # degrees


def test_set_montage_artinis_basic():
    """Test that OctaMon and Brite23 montages are set properly."""
    # Test OctaMon montage
    montage_octamon = make_standard_montage('artinis-octamon')
    montage_brite23 = make_standard_montage('artinis-brite23')
    raw = _simulate_artinis_octamon()
    raw_od = optical_density(raw)
    old_info = raw.info.copy()
    old_info_od = raw_od.info.copy()
    raw.set_montage(montage_octamon)
    raw_od.set_montage(montage_octamon)
    raw_hb = beer_lambert_law(raw_od, ppf=6)  # montage needed for BLL
    # Check that the montage was actually modified
    assert_raises(AssertionError, assert_array_almost_equal,
                  old_info['chs'][0]['loc'][:9],
                  raw.info['chs'][0]['loc'][:9])
    assert_raises(AssertionError, assert_array_almost_equal,
                  old_info_od['chs'][0]['loc'][:9],
                  raw_od.info['chs'][0]['loc'][:9])

    # Check a known location
    assert_array_almost_equal(raw.info['chs'][0]['loc'][:3],
                              [0.0616, 0.075398, 0.07347])
    assert_array_almost_equal(raw.info['chs'][8]['loc'][:3],
                              [-0.033875,  0.101276,  0.077291])
    assert_array_almost_equal(raw.info['chs'][12]['loc'][:3],
                              [-0.062749,  0.080417,  0.074884])
    assert_array_almost_equal(raw_od.info['chs'][12]['loc'][:3],
                              [-0.062749,  0.080417,  0.074884])
    assert_array_almost_equal(raw_hb.info['chs'][12]['loc'][:3],
                              [-0.062749,  0.080417,  0.074884])
    # Check that locations are identical for a pair of channels (all elements
    # except the 10th which is the wavelength if not hbo and hbr type)
    assert_array_almost_equal(raw.info['chs'][0]['loc'][:9],
                              raw.info['chs'][1]['loc'][:9])
    assert_array_almost_equal(raw_od.info['chs'][0]['loc'][:9],
                              raw_od.info['chs'][1]['loc'][:9])
    assert_array_almost_equal(raw_hb.info['chs'][0]['loc'][:9],
                              raw_hb.info['chs'][1]['loc'][:9])

    # Test Brite23 montage
    raw = _simulate_artinis_brite23()
    old_info = raw.info.copy()
    raw.set_montage(montage_brite23)
    # Check that the montage was actually modified
    assert_raises(AssertionError, assert_array_almost_equal,
                  old_info['chs'][0]['loc'][:9],
                  raw.info['chs'][0]['loc'][:9])
    # Check a known location
    assert_array_almost_equal(raw.info['chs'][0]['loc'][:3],
                              [0.085583, 0.036275, 0.089426])
    assert_array_almost_equal(raw.info['chs'][8]['loc'][:3],
                              [0.069555, 0.078579, 0.069305])
    assert_array_almost_equal(raw.info['chs'][12]['loc'][:3],
                              [0.044861, 0.100952, 0.065175])
    # Check that locations are identical for a pair of channels (all elements
    # except the 10th which is the wavelength if not hbo and hbr type)
    assert_array_almost_equal(raw.info['chs'][0]['loc'][:9],
                              raw.info['chs'][1]['loc'][:9])

    # Test channel variations
    raw_old = _simulate_artinis_brite23()
    # Raw missing some channels that are in the montage: pass
    raw = raw_old.copy()
    raw.pick(['S1_D1 hbo', 'S1_D1 hbr'])
    raw.set_montage('artinis-brite23')

    # Unconventional channel pair: pass
    raw = raw_old.copy()
    info_new = create_info(['S11_D1 hbo', 'S11_D1 hbr'], raw.info['sfreq'],
                           ['hbo', 'hbr'])
    new = RawArray(np.random.normal(size=(2, len(raw))), info_new)
    raw.add_channels([new], force_update_info=True)
    raw.set_montage('artinis-brite23')

    # Source not in montage: fail
    raw = raw_old.copy()
    info_new = create_info(['S12_D7 hbo', 'S12_D7 hbr'], raw.info['sfreq'],
                           ['hbo', 'hbr'])
    new = RawArray(np.random.normal(size=(2, len(raw))), info_new)
    raw.add_channels([new], force_update_info=True)
    with pytest.raises(ValueError, match='is not in list'):
        raw.set_montage('artinis-brite23')

    # Detector not in montage: fail
    raw = raw_old.copy()
    info_new = create_info(['S11_D8 hbo', 'S11_D8 hbr'], raw.info['sfreq'],
                           ['hbo', 'hbr'])
    new = RawArray(np.random.normal(size=(2, len(raw))), info_new)
    raw.add_channels([new], force_update_info=True)
    with pytest.raises(ValueError, match='is not in list'):
        raw.set_montage('artinis-brite23')
