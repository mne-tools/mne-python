# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

import warnings
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_allclose

from mne import write_events, read_epochs_eeglab
from mne.io import read_raw_eeglab
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.utils import _TempDir, run_tests_if_main

base_dir = op.join(testing.data_path(download=False), 'EEGLAB')
raw_fname = op.join(base_dir, 'test_raw.set')
raw_fname_onefile = op.join(base_dir, 'test_raw_onefile.set')
epochs_fname = op.join(base_dir, 'test_epochs.set')
epochs_fname_onefile = op.join(base_dir, 'test_epochs_onefile.set')
montage = op.join(base_dir, 'test_chans.locs')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


@testing.requires_testing_data
def test_io_set():
    """Test importing EEGLAB .set files"""
    _test_raw_reader(read_raw_eeglab, input_fname=raw_fname, montage=montage)
    _test_raw_reader(read_raw_eeglab, input_fname=raw_fname_onefile,
                     montage=montage)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = read_epochs_eeglab(epochs_fname)
    assert_equal(len(w), 3)

    epochs2 = read_epochs_eeglab(epochs_fname_onefile)
    assert_allclose(epochs.get_data(), epochs2.get_data(), rtol=1e-5,
                    atol=1e-5)

    temp_dir = _TempDir()
    out_fname = op.join(temp_dir, 'test-eve.fif')
    write_events(out_fname, epochs.events)
    event_id = {'S255/S8': 1, 'S8': 2, 'S255/S9': 3}

    epochs = read_epochs_eeglab(epochs_fname, epochs.events, event_id)
    epochs = read_epochs_eeglab(epochs_fname, out_fname, event_id)
    assert_raises(ValueError, read_epochs_eeglab, epochs_fname,
                  None, event_id)
    assert_raises(ValueError, read_epochs_eeglab, epochs_fname,
                  epochs.events, None)

run_tests_if_main()
