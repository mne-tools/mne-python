# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_allclose

from mne.channels import make_standard_montage
from mne._digitization.base import _get_dig_eeg
from mne._digitization._utils import _get_fid_coords
from mne.channels.montage import get_builtin_montages
from mne.io.constants import FIFF


EXPECTED_HEAD_SIZE = 0.085


@pytest.mark.parametrize('kind', get_builtin_montages())
def test_standard_montages_are_in_head_coord_or_have_fids(kind):
    """Test standard montage are all in head coord."""
    montage = make_standard_montage(kind)
    fids, coord_frame = _get_fid_coords(montage.dig)
    expected_coord = FIFF.FIFFV_COORD_UNKNOWN
    for v in fids.values():
        if v is None:
            expected_coord = FIFF.FIFFV_COORD_HEAD
            break

    for d in montage.dig:
        assert d['coord_frame'] == expected_coord


def test_standard_montage_errors():
    """Test error handling for wrong keys."""
    with pytest.raises(ValueError, match='Could not find the montage'):
        _ = make_standard_montage('not-here')


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
def test_standard_montages_on_sphere(kind, tol):
    """Test some standard montage are on sphere."""
    montage = make_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE),
        atol=tol,
    )
