# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


import pytest

import numpy as np

from numpy.testing import assert_array_equal, assert_allclose

from mne.channels.montage import read_montage
from mne.channels.montage import _BUILT_IN_MONTAGES
from mne.channels import read_standard_montage
from mne._digitization.base import _get_dig_eeg




MONTAGES_WITHOUT_FIDUCIALS = ['EGI_256', 'easycap-M1', 'easycap-M10']
MONTAGES_WITH_FIDUCIALS = [k for k in _BUILT_IN_MONTAGES
                           if k not in MONTAGES_WITHOUT_FIDUCIALS]

EXPECTED_HEAD_SIZE = 0.085


def test_read_standard_montage_egi_256():
    """Test egi_256."""
    EXPECTED_HEAD_SIZE = 0.085
    EXPECTED_HEAD_VARIANCE = 0.00418
    EXPECTED_FIRST_9_LOC = np.array(
        [[ 6.55992516e-02,  5.64176352e-02, -2.57662946e-02],  # noqa
         [ 6.08331388e-02,  6.57063949e-02, -6.40717015e-03],  # noqa
         [ 5.19851171e-02,  7.15413471e-02,  1.12091555e-02],  # noqa
         [ 4.18066179e-02,  7.31439438e-02,  2.66373224e-02],  # noqa
         [ 3.09755787e-02,  6.97928339e-02,  4.21906579e-02],  # noqa
         [ 1.96959622e-02,  6.22758709e-02,  5.58500821e-02],  # noqa
         [ 1.03933314e-02,  5.14631908e-02,  6.63221724e-02],  # noqa
         [ 8.76671630e-18,  3.81400691e-02,  7.39613137e-02],  # noqa
         [-1.05002738e-02,  1.95003515e-02,  7.85765571e-02]]  # noqa
    )

    montage = read_standard_montage('EGI_256')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])
    eeg_center = eeg_loc.mean(axis=0)
    distance_to_center = np.linalg.norm(eeg_loc - eeg_center, axis=1)

    # assert_allclose(eeg_center, [0, 0, 0], atol=1e-8)  # XXX we no longer substract mean
    assert_allclose(distance_to_center.mean(), 0.085, atol=1e-3)
    assert_allclose(distance_to_center.std(), 0.00418, atol=1e-4)
    # assert_allclose(eeg_loc[:9], EXPECTED_FIRST_9_LOC, atol=1e-1)  # XXX ?


@pytest.mark.parametrize('kind', [
    # 'EGI_256',  # This was broken
    # 'easycap-M1',  # easycap don't match.
    # 'easycap-M10',
    'GSN-HydroCel-128',
    'GSN-HydroCel-129',
    'GSN-HydroCel-256',
    'GSN-HydroCel-257',
    'GSN-HydroCel-32',
    'GSN-HydroCel-64_1.0',
    'GSN-HydroCel-65_1.0',
    'biosemi128',
    'biosemi16',
    'biosemi160',
    'biosemi256',
    'biosemi32',
    'biosemi64',
    'mgh60',
    'mgh70',
    'standard_1005',
    'standard_1020',
    'standard_alphabetic',
    'standard_postfixed',
    'standard_prefixed',
    'standard_primed',
])
def test_old_vs_new(kind):
    """Test difference between old and new standard montages."""
    mont = read_montage(kind)
    digm = read_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_standard_montage_errors():
    """Test error handling for wrong keys."""
    with pytest.raises(ValueError, match='Could not find the montage'):
        _ = read_standard_montage('not-here')


@pytest.mark.parametrize('kind, tol', [
    ['EGI_256', 1e-5],
    ['easycap-M1', 1e-8],
    ['easycap-M10', 1e-8],
])
def test_standard_montages_in_head(kind, tol):
    """Test standard montage properties (ie: they form a head)."""
    montage = read_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE),
        atol=tol,
    )
