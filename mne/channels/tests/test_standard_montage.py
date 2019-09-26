# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_allclose

from mne.channels import make_standard_montage, read_montage
from mne._digitization.base import _get_dig_eeg
from mne._digitization._utils import _get_fid_coords
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
    with pytest.raises(ValueError, match='Could not find the montage'):
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
    with pytest.deprecated_call():
        o_1005 = read_montage('standard_1005')  # old montages
        o_1020 = read_montage('standard_1020')
    # new montages, tweaked to end up at the same size as the others
    m_1005 = make_standard_montage('standard_1005', 0.0970)
    m_1020 = make_standard_montage('standard_1020', 0.0991)
    assert len(set(m_1005.ch_names) - set(m_1020.ch_names)) > 0
    # XXX weird that this is not a proper superset...
    assert set(m_1020.ch_names) - set(m_1005.ch_names) == {'O10', 'O9'}
    c_1005 = m_1005._get_ch_pos()
    for key, value in m_1020._get_ch_pos().items():
        want = o_1020.pos[o_1020.ch_names.index(key)]
        assert_allclose(value, want, err_msg=key, atol=1e-4)
        if key not in ('O10', 'O9'):
            assert_allclose(o_1005.pos[o_1005.ch_names.index(key)], want,
                            err_msg=key, atol=1e-4)
            assert_allclose(c_1005[key], value, atol=1e-4, err_msg=key)
