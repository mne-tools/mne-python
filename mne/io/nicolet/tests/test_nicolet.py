
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import numpy as np

from numpy.testing import assert_array_equal

from mne.utils import run_tests_if_main, object_diff
from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_reader
from mne.channels import Montage

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    _test_raw_reader(read_raw_nicolet, input_fname=fname, ch_type='eeg',
                     ecg='auto', eog='auto', emg='auto', misc=['PHO'])


def _fake_montage(ch_names):
    return Montage(
        pos=np.random.RandomState(42).randn(len(ch_names), 3),
        ch_names=ch_names,
        kind='foo',
        selection=np.arange(len(ch_names))
    )


def test_montage():
    """Test montage."""
    raw_none = read_raw_nicolet(input_fname=fname, ch_type='eeg',
                                montage=None, preload=False)
    montage = _fake_montage(raw_none.info['ch_names'])

    raw_montage = read_raw_nicolet(input_fname=fname, ch_type='eeg',
                                   montage=montage, preload=False)
    raw_none.set_montage(montage)

    # Check they are the same
    assert_array_equal(raw_none.get_data(), raw_montage.get_data())
    assert object_diff(raw_none.info['dig'], raw_montage.info['dig']) == ''
    assert object_diff(raw_none.info['chs'], raw_montage.info['chs']) == ''


run_tests_if_main()
