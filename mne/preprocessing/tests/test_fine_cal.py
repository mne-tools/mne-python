# Author: Mark Wronkiewicz <wronk@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
import warnings

from mne.datasets import testing
from mne.preprocessing._fine_cal import (read_fine_calibration,
                                         write_fine_calibration)
from mne.utils import _TempDir, object_hash, run_tests_if_main
from nose.tools import assert_equal

warnings.simplefilter('always')  # Always throw warnings

# Define fine calibration filepaths
data_path = testing.data_path(download=False)
fine_cal_fname = op.join(data_path, 'SSS', 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(data_path, 'SSS', 'sss_cal_3053_3d.dat')


@testing.requires_testing_data
def test_read_write_fine_cal():
    """Test round trip reading/writing of fine calibration .dat file"""
    temp_dir = _TempDir()
    temp_fname = op.join(temp_dir, 'fine_cal_temp.dat')

    for fname in [fine_cal_fname, fine_cal_fname_3d]:
        # Load fine calibration file
        fine_cal_dict = read_fine_calibration(fname)

        # Save temp version of fine calibration file
        write_fine_calibration(temp_fname, fine_cal_dict)
        fine_cal_dict_reload = read_fine_calibration(temp_fname)

        # Load temp version of fine calibration file and compare hashes
        assert_equal(object_hash(fine_cal_dict),
                     object_hash(fine_cal_dict_reload))

run_tests_if_main()
