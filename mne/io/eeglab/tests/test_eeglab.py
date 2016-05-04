# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import shutil

import warnings
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_array_equal

from mne import write_events, read_epochs_eeglab, Epochs, find_events
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
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter('always')
        # main tests, and test missing event_id
        _test_raw_reader(read_raw_eeglab, input_fname=raw_fname,
                         montage=montage)
        _test_raw_reader(read_raw_eeglab, input_fname=raw_fname_onefile,
                         montage=montage)
        assert_equal(len(w1), 20)
        # f3 or preload_false and a lot for dropping events
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # test finding events in continuous data
        event_id = {'rt': 1, 'square': 2}
        raw0 = read_raw_eeglab(input_fname=raw_fname, montage=montage,
                               event_id=event_id, preload=True)
        raw1 = read_raw_eeglab(input_fname=raw_fname, montage=montage,
                               event_id=event_id, preload=False)
        raw2 = read_raw_eeglab(input_fname=raw_fname_onefile, montage=montage,
                               event_id=event_id)
        raw3 = read_raw_eeglab(input_fname=raw_fname, montage=montage,
                               event_id=event_id)
        raw4 = read_raw_eeglab(input_fname=raw_fname, montage=montage)
        Epochs(raw0, find_events(raw0), event_id)
        epochs = Epochs(raw1, find_events(raw1), event_id)
        assert_equal(len(find_events(raw4)), 0)  # no events without event_id
        assert_equal(epochs["square"].average().nave, 80)  # 80 with
        assert_array_equal(raw0[:][0], raw1[:][0], raw2[:][0], raw3[:][0])
        assert_array_equal(raw0[:][-1], raw1[:][-1], raw2[:][-1], raw3[:][-1])
        assert_equal(len(w), 4)
        # 1 for preload=False / str with fname_onefile, 3 for dropped events
        raw0.filter(1, None)  # test that preloading works

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

    # test reading file with one event
    eeg = io.loadmat(raw_fname, struct_as_record=False,
                     squeeze_me=True)['EEG']
    one_event_fname = op.join(temp_dir, 'test_one_event.set')
    io.savemat(one_event_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': 'test_one_event.fdt',
                'epoch': eeg.epoch, 'event': eeg.event[0],
                'chanlocs': eeg.chanlocs, 'pnts': eeg.pnts}})
    shutil.copyfile(op.join(base_dir, 'test_raw.fdt'),
                    op.join(temp_dir, 'test_one_event.fdt'))
    event_id = {eeg.event[0].type: 1}
    read_raw_eeglab(input_fname=one_event_fname, montage=montage,
                    event_id=event_id, preload=True)

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
