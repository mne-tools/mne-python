# Authors: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

import os
from os import path as op
import shutil
import warnings

import numpy as np
from nose.tools import assert_raises, assert_true, assert_false
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from mne import pick_types
from mne.tests.common import assert_dig_allclose
from mne.transforms import apply_trans
from mne.io import Raw
#from mne.io import _read_mhd
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _TempDir, run_tests_if_main, slow_test
from mne.datasets import testing, spm_face

from mne.io.itab.itab import read_raw_itab


@slow_test
#@testing.requires_testing_data
def test_read_itab():
    """Test ITAB reader"""

    fname = 'd:\data\rawdata\pzzvtr62\pzzvtr62_0101.raw.mhd' 

    raw = read_raw_itab(fname, preload=True, verbose=True)
        
# 
#==============================================================================
run_tests_if_main()
