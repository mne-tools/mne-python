
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD-3-Clause

import os.path as op
import inspect

from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_reader

import pytest

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname_data = op.join(base_dir, 'test_nicolet_raw.data')
fname_head = op.join(base_dir, 'test_nicolet_raw.head')


def test_data():
    """Test reading raw nicolet files."""
    _test_raw_reader(read_raw_nicolet, input_fname=fname_data, ch_type='eeg',
                     ecg='auto', eog='auto', emg='auto', misc=['PHO'])

    with pytest.raises(ValueError,
                       match='File name should end with .data not ".head".'):
        read_raw_nicolet(fname_head, 'eeg')
