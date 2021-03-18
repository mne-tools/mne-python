# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_allclose

from mne.channels import make_standard_montage
from mne.io._digitization import _get_dig_eeg, _get_fid_coords
from mne.channels.montage import get_builtin_montages, HEAD_SIZE_DEFAULT
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


def _simulate_artinis_octomon():
    """
    Simulate artinis octomon channel data from nirx data.
    This is to test data that is imported with missing or incorrect montage
    info. This data can then me used to test the set_montage function.
    """
    import os.path as op
    from mne.datasets import fnirs_motor
    from mne.io import read_raw_nirx
    fnirs_data_folder = fnirs_motor.data_path()
    fnirs_cw_amplitude_dir = op.join(fnirs_data_folder, 'Participant-1')
    raw = read_raw_nirx(fnirs_cw_amplitude_dir, preload=True)
    mapping = {'S1_D2 760': 'S3_D1 760', 'S1_D2 850': 'S3_D1 850',
               'S1_D3 760': 'S4_D1 760', 'S1_D3 850': 'S4_D1 850',
               'S1_D9 760': 'S5_D2 760', 'S1_D9 850': 'S5_D2 850',
               'S2_D3 760': 'S6_D2 760', 'S2_D3 850': 'S6_D2 850',
               'S2_D4 760': 'S7_D2 760', 'S2_D4 850': 'S7_D2 850',
               'S2_D10 760': 'S8_D2 760', 'S2_D10 850': 'S8_D2 850'}
    raw.rename_channels(mapping)
    raw.pick(picks=["S1_D1 760", "S1_D1 850",
                    "S2_D1 760", "S2_D1 850",
                    "S3_D1 760", "S3_D1 850",
                    "S4_D1 760", "S4_D1 850",
                    "S5_D2 760", "S5_D2 850",
                    "S6_D2 760", "S6_D2 850",
                    "S7_D2 760", "S7_D2 850",
                    "S8_D2 760", "S8_D2 850"])
    return raw

def test_artinis():
    raw = _simulate_artinis_octomon()
    montage = make_standard_montage("artinis-octamon")
    raw.set_montage(montage)
