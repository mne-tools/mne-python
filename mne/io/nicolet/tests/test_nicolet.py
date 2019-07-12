
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from mne.utils import run_tests_if_main
from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_reader

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    _test_raw_reader(read_raw_nicolet, input_fname=fname, ch_type='eeg',
                     ecg='auto', eog='auto', emg='auto', misc=['PHO'])


run_tests_if_main()
