# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_allclose

from mne.channels.montage import _BUILT_IN_MONTAGES
from mne.channels import make_standard_montage
from mne._digitization.base import _get_dig_eeg
from mne.channels.montage import get_builtin_montages
from mne.io.constants import FIFF


MONTAGES_WITHOUT_FIDUCIALS = ['EGI_256', 'easycap-M1', 'easycap-M10']
MONTAGES_WITH_FIDUCIALS = [k for k in _BUILT_IN_MONTAGES
                           if k not in MONTAGES_WITHOUT_FIDUCIALS]

EXPECTED_HEAD_SIZE = 0.085


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


def _is_in_head(montage):
    for d in montage.dig:
        assert d['coord_frame'] == FIFF.FIFFV_COORD_HEAD


def test_standard_montages_in_head_coord():
    """Test standard montage are all in head coord."""
    for current_montage in get_builtin_montages():
        montage = make_standard_montage(current_montage)
        _is_in_head(montage)
