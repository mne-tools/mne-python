# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

from mne.io import read_raw_eeglab, read_epochs_eeglab
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.utils import run_tests_if_main

base_dir = op.join(testing.data_path(download=False), 'EEGLAB')
raw_fname = op.join(base_dir, 'test_raw.set')
epochs_fname = op.join(base_dir, 'test_epochs.set')
montage = op.join(base_dir, 'test_chans.locs')


@testing.requires_testing_data
def test_io_set():
    """Test importing EEGLAB .set files"""
    _test_raw_reader(read_raw_eeglab, input_fname=raw_fname, montage=montage)
    read_epochs_eeglab(epochs_fname)

run_tests_if_main()
