"""Data Equivalence Tests"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import io

from mne.utils import _TempDir
from mne.fiff import Raw, pick_types
from mne.fiff.edf import read_raw_edf

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
hpts_path = op.join(data_dir, 'biosemi.hpts')
bdf_path = op.join(data_dir, 'test.bdf')
edf_path = op.join(data_dir, 'test.edf')
bdf_eeglab_path = op.join(data_dir, 'test_bdf_eeglab.mat')
edf_eeglab_path = op.join(data_dir, 'test_edf_eeglab.mat')

tempdir = _TempDir()


def test_bdf_data():
    """Test reading raw bdf files
    """
    raw_py = read_raw_edf(bdf_path, hpts=hpts_path, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    print raw_py  # to test repr
    print raw_py.info  # to test Info repr

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(bdf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]

    assert_array_almost_equal(data_py, data_eeglab)


def test_edf_data():
    """Test reading raw edf files
    """
    raw_py = read_raw_edf(edf_path, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    print raw_py  # to test repr
    print raw_py.info  # to test Info repr

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(edf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]

    assert_array_almost_equal(data_py, data_eeglab)


def test_read_segment():
    """Test writing raw edf files when preload is False
    """
    raw1 = read_raw_edf(bdf_path, hpts=hpts_path, preload=False)
    raw1_file = op.join(tempdir, 'raw1.fif')
    raw1.save(raw1_file, overwrite=True)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:, :]
    data11, times11 = raw11[:, :]
    assert_array_almost_equal(data1, data11, 8)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    raw2 = read_raw_edf(bdf_path, hpts=hpts_path, preload=True)
    raw2_file = op.join(tempdir, 'raw2.fif')
    raw2.save(raw2_file, overwrite=True)
    data2, times2 = raw2[:, :]
    assert_array_equal(data1, data2)
    assert_array_equal(times1, times2)

    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)
