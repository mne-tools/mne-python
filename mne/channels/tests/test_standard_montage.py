# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np
import os.path as op

from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_raises)

from mne import create_info, read_trans
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.io._digitization import _get_dig_eeg, _get_fid_coords
from mne.channels.montage import get_builtin_montages, HEAD_SIZE_DEFAULT
from mne.channels import compute_native_head_t
from mne.io.constants import FIFF


@pytest.mark.parametrize('kind', get_builtin_montages())
def test_standard_montages_have_fids(kind):
    """Test standard montage are all in unknown coord (have fids)."""
    montage = make_standard_montage(kind)
    fids, coord_frame = _get_fid_coords(montage.dig)
    for k, v in fids.items():
        assert v is not None, k
    for d in montage.dig:
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
    """
    Simulate artinis octamon channel data from numpy data.
    This is to test data that is imported with missing or incorrect montage
    info. This data can then be used to test the set_montage function.
    """
    data = np.random.normal(size=(16, 100))
    ch_names = ['D1_S1 hbo', 'D1_S1 hbr', 'D1_S2 hbo', 'D1_S2 hbr',
                'D1_S3 hbo', 'D1_S3 hbr', 'D1_S4 hbo', 'D1_S4 hbr',
                'D2_S5 hbo', 'D2_S5 hbr', 'D2_S6 hbo', 'D2_S6 hbr',
                'D2_S7 hbo', 'D2_S7 hbr', 'D2_S8 hbo', 'D2_S8 hbr']
    ch_types = ['hbo', 'hbr', 'hbo', 'hbr',
                'hbo', 'hbr', 'hbo', 'hbr',
                'hbo', 'hbr', 'hbo', 'hbr',
                'hbo', 'hbr', 'hbo', 'hbr']
    sfreq = 10.  # Hz
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data, info, verbose=True)

    return raw


def test_artinis():
    raw = _simulate_artinis_octamon()
    old_info = raw.info.copy()
    montage_octamon = make_standard_montage("artinis-octamon")
    raw.set_montage(montage_octamon)
    # First check that the montage was actually modified
    assert_raises(AssertionError, assert_array_almost_equal,
                  old_info["chs"][0]["loc"][:9],
                  raw.info["chs"][0]["loc"][:9])
    # Check a known location
    assert_array_almost_equal(raw.info["chs"][0]["loc"][:3],
                              [0.0616, 0.075398, 0.07347])
    assert_array_almost_equal(raw.info["chs"][8]["loc"][:3],
                              [-0.033875,  0.101276,  0.077291])
    assert_array_almost_equal(raw.info["chs"][12]["loc"][:3],
                              [-0.062749,  0.080417,  0.074884])
    # fNIRS has two identical channel locations for each measurement
    # The 10th element encodes the wavelength, so it will differ.
    assert_array_almost_equal(raw.info["chs"][0]["loc"][:9],
                              raw.info["chs"][1]["loc"][:9])
    # Compare OctaMon and Brite23 to fsaverage
    trans_octamon = compute_native_head_t(montage_octamon)
    montage_brite = make_standard_montage("artinis-brite23")
    trans_brite = compute_native_head_t(montage_brite)
    fif = op.join(op.dirname(__file__), '..', '..', 'data', 'fsaverage',
                  'fsaverage-trans.fif')
    fsaverage = read_trans(fif)
    assert_array_almost_equal(list(trans_octamon.values())[2],
                              list(fsaverage.values())[2])
    assert_array_almost_equal(list(trans_brite.values())[2],
                              list(fsaverage.values())[2])
