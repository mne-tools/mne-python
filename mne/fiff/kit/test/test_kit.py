"""Data and Channel Location Equivalence Tests"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect
from numpy.testing import assert_array_almost_equal, assert_array_equal
import scipy.io
from mne.fiff import Raw, pick_types, kit
from mne.utils import _TempDir

FILE = inspect.getfile(inspect.currentframe())
parent_dir = op.dirname(op.abspath(FILE))
data_dir = op.join(parent_dir, 'data')
tempdir = _TempDir()


def test_data():
    """Test reading raw kit files
    """
    raw_py = kit.read_raw_kit(input_fname=op.join(data_dir, 'test.sqd'),
                           mrk_fname=op.join(data_dir, 'test_mrk.sqd'),
                           elp_fname=op.join(data_dir, 'test_elp.txt'),
                           hsp_fname=op.join(data_dir, 'test_hsp.txt'),
                           sns_fname=op.join(data_dir, 'sns.txt'),
                           stim=range(167, 159, -1), preload=True)
    # Binary file only stores the sensor channels
    py_picks = pick_types(raw_py.info, exclude=[])
    raw_bin = op.join(data_dir, 'test_bin.fif')
    raw_bin = Raw(raw_bin, preload=True)
    bin_picks = pick_types(raw_bin.info, exclude=[])
    data_bin, _ = raw_bin[bin_picks]
    data_py, _ = raw_py[py_picks]

    # this .mat was generated using the Yokogawa MEG Reader
    data_Ykgw = op.join(data_dir, 'test_Ykgw.mat')
    data_Ykgw = scipy.io.loadmat(data_Ykgw)['data']
    data_Ykgw = data_Ykgw[py_picks]

    assert_array_almost_equal(data_py, data_Ykgw)

    data_py = data_py[bin_picks]
    assert_array_almost_equal(data_py, data_bin)


def test_read_segment():
    """Test writing raw kit files when preload is False
    """
    raw1 = kit.read_raw_kit(input_fname=op.join(data_dir, 'test.sqd'),
                           mrk_fname=op.join(data_dir, 'test_mrk.sqd'),
                           elp_fname=op.join(data_dir, 'test_elp.txt'),
                           hsp_fname=op.join(data_dir, 'test_hsp.txt'),
                           sns_fname=op.join(data_dir, 'sns.txt'),
                           stim=range(167, 159, -1), preload=False)
    raw1_file = op.join(tempdir, 'raw1.fif')
    raw1.save(raw1_file, buffer_size_sec=.1, overwrite=True)
    raw2 = kit.read_raw_kit(input_fname=op.join(data_dir, 'test.sqd'),
                           mrk_fname=op.join(data_dir, 'test_mrk.sqd'),
                           elp_fname=op.join(data_dir, 'test_elp.txt'),
                           hsp_fname=op.join(data_dir, 'test_hsp.txt'),
                           sns_fname=op.join(data_dir, 'sns.txt'),
                           stim=range(167, 159, -1), preload=True)
    raw2_file = op.join(tempdir, 'raw2.fif')
    raw2.save(raw2_file, buffer_size_sec=.1, overwrite=True)
    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)


def test_ch_loc():
    """Test raw kit loc
    """
    raw_py = kit.read_raw_kit(input_fname=op.join(data_dir, 'test.sqd'),
                       mrk_fname=op.join(data_dir, 'test_mrk.sqd'),
                       elp_fname=op.join(data_dir, 'test_elp.txt'),
                       hsp_fname=op.join(data_dir, 'test_hsp.txt'),
                       sns_fname=op.join(data_dir, 'sns.txt'),
                       stim=range(167, 159, -1))
    raw_bin = Raw(op.join(data_dir, 'test_bin.fif'))

    for py_ch, bin_ch in zip(raw_py.info['chs'], raw_bin.info['chs']):
        if bin_ch['ch_name'].startswith('MEG'):
            # the mne_kit2fiff_bin has a different representation of pi.
            assert_array_almost_equal(py_ch['loc'], bin_ch['loc'], decimal=5)
