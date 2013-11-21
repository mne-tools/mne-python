"""Data Equivalence Tests"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.utils import _TempDir
from mne.fiff import Raw, pick_types
from mne.fiff.brainvision import read_raw_brainvision

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
vhdr_path = op.join(data_dir, 'test.vhdr')
elp_path = op.join(data_dir, 'test_elp.txt')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
ch_names = ['FP1', 'VEOGt', 'F7', 'GND', 'F8',
            'FC5', 'F3', 'FZ', 'F4', 'FC6',
            'FC1', 'FCZ', 'FC2', 'CP5', 'C3',
            'CZ', 'C4', 'CP6', 'CP1', 'CPZ',
            'CP2', 'P7', 'P3', 'PZ', 'P4',
            'P8', 'O1', 'POZ', 'O2', 'A1',
            'A2', 'HEOGL', 'HEOGR', 'VEOGb']

tempdir = _TempDir()


def test_brainvision_data():
    """Test reading raw Brain Vision files
    """
    raw_py = read_raw_brainvision(vhdr_path, elp_fname=elp_path,
                                  ch_names=ch_names, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, times_py = raw_py[picks]

    print raw_py  # to test repr
    print raw_py.info  # to test Info repr

    # this fif was generated using MNE-C
    raw_bin = Raw(eeg_bin, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_bin, times_bin = raw_bin[picks]

    assert_array_almost_equal(data_py, data_bin)
    assert_array_almost_equal(times_py, times_bin)


def test_read_segment():
    """Test writing raw eeg files when preload is False
    """
    raw1 = read_raw_brainvision(vhdr_path, preload=False)
    raw1_file = op.join(tempdir, 'raw1.fif')
    raw1.save(raw1_file, overwrite=True)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:, :]
    data11, times11 = raw11[:, :]
    assert_array_almost_equal(data1, data11, 8)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    raw2 = read_raw_brainvision(vhdr_path, preload=True)
    raw2_file = op.join(tempdir, 'raw2.fif')
    raw2.save(raw2_file, overwrite=True)
    data2, times2 = raw2[:, :]
    assert_array_equal(data1, data2)
    assert_array_equal(times1, times2)

    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)
