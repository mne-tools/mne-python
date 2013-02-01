"""Data Equivalence Test"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from numpy.testing import assert_array_almost_equal
import mne
import mne.fiff.kit as kit
import scipy.io
import inspect
import os

parent = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(parent, 'data')

raw_py = kit.read_raw_kit(input_fname=os.path.join(data_dir, 'test.sqd'),
                       mrk_fname=os.path.join(data_dir, 'test_marker.txt'),
                       elp_fname=os.path.join(data_dir, 'test.elp'),
                       hsp_fname=os.path.join(data_dir, 'test.hsp'),
                       sns_fname=os.path.join(data_dir, 'sns.txt'))
#last row is the synthetic trigger channel that is created within module
data_py = raw_py._data[:-1, :]
#this .mat was generated using the Yokogawa MEG Reader
data_Ykgw = scipy.io.loadmat(os.path.join(data_dir, 'test_Ykgw.mat'))['data']

assert_array_almost_equal(data_py, data_Ykgw)

# Binary file only stores the sensor channels
raw_bin = mne.fiff.Raw(os.path.join(data_dir, 'test_bin.fif'), preload=True)
data_bin = raw_bin._data[:157, :]
data_py = data_py[:157, :]

assert_array_almost_equal(data_py, data_bin)
