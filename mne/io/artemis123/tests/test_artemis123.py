
# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect
from mne.utils import run_tests_if_main

from mne.io import read_raw_artemis123
from mne.io.tests.test_raw import _test_raw_reader

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
fname = op.join(base_dir, 'test_nicolet_raw.data')


def test_data():
    """Test reading raw Artemis123 files."""
    _test_raw_reader(read_raw_artemis123, input_fname=fname)
    pass

run_tests_if_main()
