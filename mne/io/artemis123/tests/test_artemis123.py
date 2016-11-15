
# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD (3-clause)

import os
import os.path as op
import tempfile
from numpy.testing import assert_equal

from mne.utils import run_tests_if_main
from mne.io import read_raw_artemis123
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.io.artemis123.utils import _generate_mne_locs_file, _load_mne_locs

artemis123_dir = op.join(testing.data_path(download=False), 'ARTEMIS123')
short_no_HPI_fname = op.join(artemis123_dir,
                             'Artemis_Data_2016-11-03-15h-58m_test.bin')


@testing.requires_testing_data
def test_data():
    """Test reading raw Artemis123 files."""
    _test_raw_reader(read_raw_artemis123, input_fname=short_no_HPI_fname)


def test_utils():
    """Test artemis123 utils."""
    # make a tempfile
    fd, path = tempfile.mkstemp()
    try:
        _generate_mne_locs_file(path)
        installed_locs = _load_mne_locs()
        generated_locs = _load_mne_locs(path)
        assert_equal(installed_locs, generated_locs)
    finally:
        os.remove(path)


run_tests_if_main()
