# -*- coding: utf-8 -*-
# Authors: Robert Luke  <mail@robertluke.net>
#          simplified BSD-3 license
import os.path as op

from numpy.testing import assert_array_equal
from scipy import io as sio

from mne.io import read_raw_eximia
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data


@requires_testing_data
def test_nirx():
    """Test reading NIRX files."""


run_tests_if_main()
