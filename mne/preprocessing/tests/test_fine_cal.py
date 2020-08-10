# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import pytest

from mne.datasets import testing
from mne.preprocessing import read_fine_calibration, write_fine_calibration
from mne.utils import object_diff

# Define fine calibration filepaths
data_path = testing.data_path(download=False)
fine_cal_fname = op.join(data_path, 'SSS', 'sss_cal_3053.dat')
fine_cal_fname_3d = op.join(data_path, 'SSS', 'sss_cal_3053_3d.dat')


@pytest.mark.parametrize('fname', (fine_cal_fname, fine_cal_fname_3d))
@testing.requires_testing_data
def test_read_write_fine_cal(tmpdir, fname):
    """Test round trip reading/writing of fine calibration .dat file."""
    temp_fname = op.join(str(tmpdir), 'fine_cal_temp.dat')
    # Load fine calibration file
    fine_cal_dict = read_fine_calibration(fname)

    # Save temp version of fine calibration file
    write_fine_calibration(temp_fname, fine_cal_dict)
    fine_cal_dict_reload = read_fine_calibration(temp_fname)

    # Load temp version of fine calibration file and compare hashes
    assert object_diff(fine_cal_dict, fine_cal_dict_reload) == ''
