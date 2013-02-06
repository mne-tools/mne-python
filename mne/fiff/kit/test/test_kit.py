"""Data Equivalence Test"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from numpy.testing import assert_array_almost_equal
import mne
import mne.fiff.kit as kit
from mne.fiff.kit.constants import KIT
import scipy.io
import inspect
import os

FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')


def test_data():
    raw_py = kit.read_raw_kit(input_fname=os.path.join(data_dir, 'test.sqd'),
                           mrk_fname=os.path.join(data_dir, 'test_marker.txt'),
                           elp_fname=os.path.join(data_dir, 'test_elp.txt'),
                           hsp_fname=os.path.join(data_dir, 'test_hsp.txt'),
                           sns_fname=os.path.join(data_dir, 'sns.txt'))
    #last row is the synthetic trigger channel that is created within module
    data_py = raw_py._data[:-1, :]
    #this .mat was generated using the Yokogawa MEG Reader
    data_Ykgw = os.path.join(data_dir, 'test_Ykgw.mat')
    data_Ykgw = scipy.io.loadmat(data_Ykgw)['data']

    assert_array_almost_equal(data_py, data_Ykgw)

    # Binary file only stores the sensor channels
    raw_bin = os.path.join(data_dir, 'test_bin.fif')
    raw_bin = mne.fiff.Raw(raw_bin, preload=True)
    data_bin = raw_bin._data[:KIT.n_sens, :]
    data_py = data_py[:KIT.n_sens, :]

    assert_array_almost_equal(data_py, data_bin)


def test_ch_loc():
    raw_py = kit.read_raw_kit(input_fname=os.path.join(data_dir, 'test.sqd'),
                       mrk_fname=os.path.join(data_dir, 'test_marker.txt'),
                       elp_fname=os.path.join(data_dir, 'test_elp.txt'),
                       hsp_fname=os.path.join(data_dir, 'test_hsp.txt'),
                       sns_fname=os.path.join(data_dir, 'sns.txt'))
    raw_bin = mne.fiff.Raw(os.path.join(data_dir, 'test_bin.fif'))

    for py_ch, bin_ch in zip(raw_py.info['chs'], raw_bin.info['chs']):
        if py_ch['ch_name'].startswith('MEG'):
            # the mne_kit2fiff_bin has a different representation of pi.
            assert_array_almost_equal(py_ch['loc'], bin_ch['loc'], decimal=5)
