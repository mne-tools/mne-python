# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import shutil

import warnings
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_array_equal

from mne import write_events, read_epochs_eeglab
from mne.io import read_raw_eeglab
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.utils import _TempDir, run_tests_if_main, requires_version

base_dir = op.join(testing.data_path(download=False), 'EEGLAB')
raw_fname = op.join(base_dir, 'test_raw.set')
raw_fname_onefile = op.join(base_dir, 'test_raw_onefile.set')
epochs_fname = op.join(base_dir, 'test_epochs.set')
epochs_fname_onefile = op.join(base_dir, 'test_epochs_onefile.set')
montage = op.join(base_dir, 'test_chans.locs')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


@requires_version('scipy', '0.12')
@testing.requires_testing_data
def test_io_set():
    """Test importing EEGLAB .set files"""
    from scipy import io

    _test_raw_reader(read_raw_eeglab, input_fname=raw_fname, montage=montage)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _test_raw_reader(read_raw_eeglab, input_fname=raw_fname_onefile,
                         montage=montage)
        raw = read_raw_eeglab(input_fname=raw_fname_onefile, montage=montage)
        raw2 = read_raw_eeglab(input_fname=raw_fname, montage=montage)
        assert_array_equal(raw[:][0], raw2[:][0])
    # one warning per each preload=False or str with raw_fname_onefile
    assert_equal(len(w), 3)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        epochs = read_epochs_eeglab(epochs_fname)
        epochs2 = read_epochs_eeglab(epochs_fname_onefile)
    # 3 warnings for each read_epochs_eeglab because there are 3 epochs
    # associated with multiple events
    assert_equal(len(w), 6)
    assert_array_equal(epochs.get_data(), epochs2.get_data())

    # test different combinations of events and event_ids
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

    # test if .dat file raises an error
    eeg = io.loadmat(epochs_fname, struct_as_record=False,
                     squeeze_me=True)['EEG']
    eeg.data = 'epochs_fname.dat'
    bad_epochs_fname = op.join(temp_dir, 'test_epochs.set')
    io.savemat(bad_epochs_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': eeg.data,
                'epoch': eeg.epoch, 'event': eeg.event,
                'chanlocs': eeg.chanlocs}})
    shutil.copyfile(op.join(base_dir, 'test_epochs.fdt'),
                    op.join(temp_dir, 'test_epochs.dat'))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert_raises(NotImplementedError, read_epochs_eeglab,
                      bad_epochs_fname)
    assert_equal(len(w), 3)

run_tests_if_main()
