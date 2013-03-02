"""Data and Channel Location Equivalence Tests"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
import inspect
from numpy.testing import assert_array_equal
from mne.fiff import Raw, kit

FILE = inspect.getfile(inspect.currentframe())
parent_dir = os.path.dirname(os.path.abspath(FILE))
data_dir = os.path.join(parent_dir, 'data')


raw1 = kit.read_raw_kit(input_fname=os.path.join(data_dir, 'test.sqd'),
                       mrk_fname=os.path.join(data_dir, 'test_mrk.sqd'),
                       elp_fname=os.path.join(data_dir, 'test_elp.txt'),
                       hsp_fname=os.path.join(data_dir, 'test_hsp.txt'),
                       sns_fname=os.path.join(data_dir, 'sns.txt'),
                       stim=range(167, 159, -1), preload=False)
raw1.save('/Users/teon/Desktop/raw1.fif', buffer_size_sec=.1, overwrite=True)
raw2 = kit.read_raw_kit(input_fname=os.path.join(data_dir, 'test.sqd'),
                       mrk_fname=os.path.join(data_dir, 'test_mrk.sqd'),
                       elp_fname=os.path.join(data_dir, 'test_elp.txt'),
                       hsp_fname=os.path.join(data_dir, 'test_hsp.txt'),
                       sns_fname=os.path.join(data_dir, 'sns.txt'),
                       stim=range(167, 159, -1), preload=True)
raw2.save('/Users/teon/Desktop/raw2.fif', overwrite=True)
raw1 = Raw('/Users/teon/Desktop/raw1.fif', preload=True)
raw2 = Raw('/Users/teon/Desktop/raw2.fif', preload=True)
assert_array_equal(raw1._data, raw2._data)
