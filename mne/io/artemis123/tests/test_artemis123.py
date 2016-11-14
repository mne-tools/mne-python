
# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import os.path as op

from mne.utils import run_tests_if_main
from mne.io import read_raw_artemis123
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing

artemis123_dir = op.join(testing.data_path(download=False), 'ARTEMIS123')
short_no_HPI_fname = op.join(artemis123_dir,'Artemis_Data_2016-11-03-15h-58m_test.bin')

def test_data():
    """Test reading raw Artemis123 files."""
    _test_raw_reader(read_raw_artemis123, input_fname=short_no_HPI_fname)
    pass

run_tests_if_main()
