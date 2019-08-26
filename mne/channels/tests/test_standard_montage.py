# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# import os
# import os.path as op

import pytest

# import numpy as np
# from scipy.io import savemat
# from copy import deepcopy
# from functools import partial

# from numpy.testing import (assert_array_equal, assert_almost_equal,
#                            assert_allclose, assert_array_almost_equal,
#                            assert_array_less, assert_equal)

# from mne import create_info, EvokedArray, read_evokeds, __file__ as _mne_file
# from mne.channels import (Montage, read_montage, read_dig_montage,
#                           get_builtin_montages, DigMontage,
#                           read_dig_egi, read_dig_captrack, read_dig_fif)
from mne.channels.montage import read_montage
from mne.channels.montage import _BUILT_IN_MONTAGES
# from mne.channels.montage import transform_to_head
# from mne.channels._dig_montage_utils import _transform_to_head_call
# from mne.channels._dig_montage_utils import _fix_data_fiducials
from mne.channels._standard_montage_utils import read_standard_montage
# from mne.utils import (_TempDir, run_tests_if_main, assert_dig_allclose,
#                        object_diff, Bunch)
# from mne.bem import _fit_sphere
# from mne.transforms import apply_trans, get_ras_to_neuromag_trans
# from mne.io.constants import FIFF
# from mne._digitization import Digitization
# from mne._digitization._utils import _read_dig_points
# from mne.viz._3d import _fiducial_coords

# from mne.io.kit import read_mrk
# from mne.io import (read_raw_brainvision, read_raw_egi, read_raw_fif,
#                     read_raw_cnt, read_raw_edf, read_raw_nicolet,
#                     read_raw_bdf, read_raw_eeglab, read_fiducials,
#                     __file__ as _mne_io_file)

# from mne.datasets import testing

# from mock import patch

from unittest.mock import patch


@pytest.mark.parametrize('kind', _BUILT_IN_MONTAGES)
def test_read_montage(kind):
    """Test difference between old and new standard montages."""
    print(kind)
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    # assert object_diff(new_montage, old_montage) == ''
    # assert new_montage == old_montage
    assert new_montage.__repr__() == old_montage.__repr__()


def _custom_compare_true(self, other):
    return True


def _custom_compare_false(self, other):
    return False


@pytest.mark.parametrize('kind', _BUILT_IN_MONTAGES)
@patch("mne.channels.Montage.__eq__", _custom_compare_true)
def test_read_montage_TRUE(kind):
    """Test difference between old and new standard montages."""
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    assert new_montage == old_montage


@pytest.mark.parametrize('kind', _BUILT_IN_MONTAGES)
@patch("mne.channels.Montage.__eq__", _custom_compare_false)
def test_read_montage_FALSE(kind):
    """Test difference between old and new standard montages."""
    old_montage = read_montage(kind)
    new_montage = read_standard_montage(kind)
    assert new_montage != old_montage
