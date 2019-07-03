
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import pytest


from mne.utils import run_tests_if_main
from mne.io import read_raw_nicolet
from mne.io.tests.test_raw import _test_raw_reader

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw nicolet files."""
    with pytest.deprecated_call():
        _test_raw_reader(read_raw_nicolet, input_fname=fname, ch_type='eeg',
                         ecg='auto', eog='auto', emg='auto', misc=['PHO'])


def test_montage_deprecation():
    """Test deprecation."""
    EXPECTED_DEPRECATION_MESSAGE = (
        '`montage` is deprecated since 0.19 and will be removed in 0.20.'
    )
    read_raw_nicolet(input_fname=fname, ch_type='eeg')
    with pytest.deprecated_call() as recwarn:
        read_raw_nicolet(input_fname=fname, ch_type='eeg', montage=None)
    assert len(recwarn) == 1
    assert recwarn[0].message.args[0] == EXPECTED_DEPRECATION_MESSAGE


run_tests_if_main()
