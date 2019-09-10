# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# import os
# import os.path as op

import pytest
from pytest import approx

import numpy as np
# from scipy.io import savemat
# from copy import deepcopy
# from functools import partial

from numpy.testing import assert_array_equal, assert_allclose

# from mne import create_info, EvokedArray, read_evokeds, __file__ as _mne_file
# from mne.channels import (Montage, read_montage, read_dig_montage,
#                           get_builtin_montages, DigMontage,
#                           read_dig_egi, read_dig_captrack, read_dig_fif)
from mne.channels.montage import read_montage
from mne.channels.montage import Montage, DigMontage
from mne.channels.montage import _BUILT_IN_MONTAGES
# from mne.channels.montage import transform_to_head
# from mne.channels._dig_montage_utils import _transform_to_head_call
# from mne.channels._dig_montage_utils import _fix_data_fiducials
from mne.channels._standard_montage_utils import read_standard_montage
from mne.utils import Bunch
# from mne.utils import (_TempDir, run_tests_if_main, assert_dig_allclose,
#                        object_diff, Bunch)
# from mne.bem import _fit_sphere
# from mne.transforms import apply_trans, get_ras_to_neuromag_trans
# from mne.io.constants import FIFF
from mne._digitization import Digitization
from mne._digitization.base import _get_dig_eeg
# from mne._digitization._utils import _read_dig_points
# from mne.viz._3d import _fiducial_coords

# from mne.io.kit import read_mrk
# from mne.io import (read_raw_brainvision, read_raw_egi, read_raw_fif,
#                     read_raw_cnt, read_raw_edf, read_raw_nicolet,
#                     read_raw_bdf, read_raw_eeglab, read_fiducials,
#                     __file__ as _mne_io_file)

from mne.io import RawArray
from mne import create_info
# from mne.datasets import testing

# from mock import patch

from unittest.mock import patch
from mne.io.constants import FIFF
# from mne.channels._dig_montage_utils import _get_fid_coords
from mne.channels._dig_montage_utils import _cardinal_ident_mapping

MONTAGES_WITHOUT_FIDUCIALS = ['EGI_256', 'easycap-M1', 'easycap-M10']
MONTAGES_WITH_FIDUCIALS = [k for k in _BUILT_IN_MONTAGES
                           if k not in MONTAGES_WITHOUT_FIDUCIALS]


# XXX: this should go in _digitization/utils
def _get_ch_pos_location(dig):
    return [d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_EEG]


# XXX: this should go in _digitization/utils
# XXX: this is really similar to _get_fid_coords from pr-6706 but I needed
#      something different so, I'll merge later
def _get_fid_coords(dig):
    fid_coords = Bunch(nasion=None, lpa=None, rpa=None)
    fid_coord_frames = Bunch(nasion=None, lpa=None, rpa=None)

    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            key = _cardinal_ident_mapping[d['ident']]
            fid_coords[key] = d['r']
            fid_coord_frames[key] = d['coord_frame']

    return fid_coords, fid_coord_frames


def _compare_dig_montage_and_standard_montage(self, other):
    """Allow ACTUAL_DigMontage == EXPECTED_Montage."""
    assert isinstance(self, DigMontage), 'DigMontage should be left element'
    assert isinstance(other, Montage), 'Montage should be right element'

    assert len(self.ch_names) == len(other.ch_names)

    dig_montage_fid, _ = _get_fid_coords(self.dig)
    assert dig_montage_fid.nasion == other.nasion
    assert dig_montage_fid.lpa == other.lpa
    assert dig_montage_fid.rpa == other.rpa

    dig_montage_ch_pos = dict(zip(
        self.ch_names, _get_ch_pos_location(self.dig)))
    montage_ch_pos = dict(zip(other.ch_names, other.pos))
    for kk, expected_pos in montage_ch_pos.items():
        assert_array_equal(dig_montage_ch_pos[kk], expected_pos)

    return True  # If all assert pass, then they are equal


def _compare_dig_montage_and_standard_montage_(self, other):
    """Allow ACTUAL_DigMontage == EXPECTED_Montage."""
    assert isinstance(self, DigMontage), 'DigMontage should be left element'
    assert isinstance(other, Montage), 'Montage should be right element'

    assert len(self.ch_names) == len(other.ch_names)

    dig_montage_fid, _ = _get_fid_coords(self.dig)
    montage_fid = dict(zip(
        ['nasion', 'lpa', 'rpa'], [other.nasion, other.lpa, other.rpa]
    ))

    for kk in dig_montage_fid.keys():
        assert dig_montage_fid[kk] is not None
        assert montage_fid[kk] is not None
        assert_array_equal(dig_montage_fid[kk], montage_fid[kk])

    dig_montage_ch_pos = dict(zip(
        self.ch_names, _get_ch_pos_location(self.dig)))
    montage_ch_pos = dict(zip(other.ch_names, other.pos))
    for kk, expected_pos in montage_ch_pos.items():
        assert_array_equal(dig_montage_ch_pos[kk], expected_pos)

    return True  # If all assert pass, then they are equal


@pytest.mark.skip(reason="This is no longer valid")
# @pytest.mark.parametrize('kind', _BUILT_IN_MONTAGES)
# @pytest.mark.parametrize('kind', [_BUILT_IN_MONTAGES[0]])
# @pytest.mark.parametrize('kind', MONTAGES_WITHOUT_FIDUCIALS)
@pytest.mark.parametrize('kind', [MONTAGES_WITHOUT_FIDUCIALS[1]])
@patch("mne.channels.DigMontage.__eq__",
       _compare_dig_montage_and_standard_montage)
def test_no_fid_read_montage(kind):
    """Test difference between old and new standard montages."""
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    assert new_montage == old_montage

@pytest.mark.skip(reason="This is no longer valid")
@pytest.mark.parametrize('kind', MONTAGES_WITH_FIDUCIALS)
@patch("mne.channels.DigMontage.__eq__",
       _compare_dig_montage_and_standard_montage_)
def test_read_montage(kind):
    """Test difference between old and new standard montages."""
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    assert new_montage == old_montage

    raw = RawArray(
        data=np.empty((len(old_montage.ch_names), 1), dtype=np.float64),
        info=create_info(
            ch_names=old_montage.ch_names, sfreq=1., ch_types='eeg'
        )
    ).set_montage(old_montage)

    dig = raw.info['dig']
    print(dig)
    # pass


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
    'easycap-M1',
    'easycap-M10',
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
def test_foo(kind):
    """Test difference between old and new standard montages."""
    mont = read_montage(kind)
    digm = read_standard_montage(kind)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])

    # import pdb; pdb.set_trace()
    # assert new_montage == old_montage

    ## This wont work because they are not in head
    # assert_allclose(
    #     actual=np.linalg.norm(eeg_loc, axis=1),
    #     desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE)
    # )


def test_standard_1005():
    """Test difference between old and new standard montages."""
    mont = read_montage('standard_1005')
    digm = read_standard_montage('standard_1005')
    assert isinstance(digm, DigMontage)
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_mgh60():
    """Test difference between old and new standard montages."""
    mont = read_montage('mgh60')
    digm = read_standard_montage('mgh60')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_biosemi128():
    """Test difference between old and new standard montages."""
    mont = read_montage('biosemi128')
    digm = read_standard_montage('biosemi128')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_hydrocell_128():
    """Test difference between old and new standard montages."""
    mont = read_montage('GSN-HydroCel-128')
    digm = read_standard_montage('GSN-HydroCel-128')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_hydrocell_129():
    """Test difference between old and new standard montages."""
    mont = read_montage('GSN-HydroCel-128')
    digm = read_standard_montage('GSN-HydroCel-128')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(digm.dig)])

    # Assert we are reading the same thing. (notice dig reorders chnames)
    actual = dict(zip(digm.ch_names, eeg_loc))
    expected = dict(zip(mont.ch_names, mont.pos))
    for kk in actual:
        assert_array_equal(actual[kk], expected[kk])


def test_easycap_M1():
    """Test easycap_M1."""
    EXPECTED_HEAD_SIZE = 0.085

    montage = read_standard_montage('easycap-M1')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE)
    )


def test_easycap_M10():
    """Test easycap_M1."""
    EXPECTED_HEAD_SIZE = 0.085

    montage = read_standard_montage('easycap-M10')
    eeg_loc = np.array([ch['r'] for ch in _get_dig_eeg(montage.dig)])

    assert_allclose(
        actual=np.linalg.norm(eeg_loc, axis=1),
        desired=np.full((eeg_loc.shape[0], ), EXPECTED_HEAD_SIZE)
    )
