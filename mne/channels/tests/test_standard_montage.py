# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# import os
# import os.path as op

import pytest

import numpy as np
# from scipy.io import savemat
# from copy import deepcopy
# from functools import partial

from numpy.testing import assert_array_equal

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


# @pytest.mark.parametrize('kind', _BUILT_IN_MONTAGES)
# @pytest.mark.parametrize('kind', [_BUILT_IN_MONTAGES[0]])
# @pytest.mark.parametrize('kind', MONTAGES_WITHOUT_FIDUCIALS)
@pytest.mark.parametrize('kind', [MONTAGES_WITHOUT_FIDUCIALS[0]])
@patch("mne.channels.DigMontage.__eq__",
       _compare_dig_montage_and_standard_montage)
def test_no_fid_read_montage(kind):
    """Test difference between old and new standard montages."""
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    assert new_montage == old_montage


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
    new_montage = read_standard_montage('EGI_256')

