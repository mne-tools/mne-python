# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#         Mikolaj Magnuski <mmagnuski@swps.edu.pl>
#
# License: BSD (3-clause)

from copy import deepcopy
from distutils.version import LooseVersion
import os.path as op
import shutil
from unittest import SkipTest

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal)
import pytest
from scipy import io

from mne import write_events, read_epochs_eeglab, Epochs, find_events
from mne.io import read_raw_eeglab
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.eeglab.eeglab import read_events_eeglab
from mne.io.eeglab import read_annotations_eeglab
from mne.datasets import testing
from mne.utils import run_tests_if_main, requires_h5py

base_dir = op.join(testing.data_path(download=False), 'EEGLAB')

raw_fname_mat = op.join(base_dir, 'test_raw.set')
raw_fname_onefile_mat = op.join(base_dir, 'test_raw_onefile.set')
epochs_fname_mat = op.join(base_dir, 'test_epochs.set')
epochs_fname_onefile_mat = op.join(base_dir, 'test_epochs_onefile.set')
raw_mat_fnames = [raw_fname_mat, raw_fname_onefile_mat]
epochs_mat_fnames = [epochs_fname_mat, epochs_fname_onefile_mat]

raw_fname_h5 = op.join(base_dir, 'test_raw_h5.set')
raw_fname_onefile_h5 = op.join(base_dir, 'test_raw_onefile_h5.set')
epochs_fname_h5 = op.join(base_dir, 'test_epochs_h5.set')
epochs_fname_onefile_h5 = op.join(base_dir, 'test_epochs_onefile_h5.set')
raw_h5_fnames = [raw_fname_h5, raw_fname_onefile_h5]
epochs_h5_fnames = [epochs_fname_h5, epochs_fname_onefile_h5]

raw_fnames = [raw_fname_mat, raw_fname_onefile_mat,
              raw_fname_h5, raw_fname_onefile_h5]
montage = op.join(base_dir, 'test_chans.locs')


def _check_h5(fname):
    if fname.endswith('_h5.set'):
        try:
            import h5py  # noqa, analysis:ignore
        except Exception:
            raise SkipTest('h5py module required')


@requires_h5py
@testing.requires_testing_data
@pytest.mark.parametrize('fnames', [raw_mat_fnames, raw_h5_fnames])
def test_io_set_raw(fnames, tmpdir):
    """Test importing EEGLAB .set files."""
    tmpdir = str(tmpdir)
    raw_fname, raw_fname_onefile = fnames
    with pytest.warns(RuntimeWarning) as w:
        _test_raw_reader(read_raw_eeglab, input_fname=raw_fname,
                         montage=montage)
        _test_raw_reader(read_raw_eeglab, input_fname=raw_fname_onefile,
                         montage=montage)
    for want in ('Events like', 'consist entirely', 'could not be mapped',
                 'string preload is not supported'):
        assert (any(want in str(ww.message) for ww in w))
    with pytest.warns(RuntimeWarning) as w:
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
        raw0.filter(1, None, l_trans_bandwidth='auto', filter_length='auto',
                    phase='zero')  # test that preloading works

    # test that using uint16_codec does not break stuff
    raw0 = read_raw_eeglab(input_fname=raw_fname, montage=montage,
                           event_id=event_id, preload=False,
                           uint16_codec='ascii')

    # test old EEGLAB version event import (read old version)
    eeg = io.loadmat(raw_fname_mat, struct_as_record=False,
                     squeeze_me=True)['EEG']
    for event in eeg.event:  # old version allows integer events
        event.type = 1
    assert_equal(read_events_eeglab(eeg)[-1, -1], 1)
    eeg.event = eeg.event[0]  # single event
    assert_equal(read_events_eeglab(eeg)[-1, -1], 1)

    # test reading file with one event (read old version)
    eeg = io.loadmat(raw_fname_mat, struct_as_record=False,
                     squeeze_me=True)['EEG']
    one_event_fname = op.join(tmpdir, 'test_one_event.set')
    io.savemat(one_event_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': 'test_one_event.fdt',
                'epoch': eeg.epoch, 'event': eeg.event[0],
                'chanlocs': eeg.chanlocs, 'pnts': eeg.pnts}},
               appendmat=False, oned_as='row')
    shutil.copyfile(op.join(base_dir, 'test_raw.fdt'),
                    one_event_fname.replace('.set', '.fdt'))
    event_id = {eeg.event[0].type: 1}
    test_raw = read_raw_eeglab(input_fname=one_event_fname, montage=montage,
                               event_id=event_id, preload=True)

    # test that sample indices are read python-wise (zero-based)
    assert find_events(test_raw)[0, 0] == round(eeg.event[0].latency) - 1

    # test negative event latencies
    negative_latency_fname = op.join(tmpdir, 'test_negative_latency.set')
    evnts = deepcopy(eeg.event[0])
    evnts.latency = 0
    io.savemat(negative_latency_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': 'test_one_event.fdt',
                'epoch': eeg.epoch, 'event': evnts,
                'chanlocs': eeg.chanlocs, 'pnts': eeg.pnts}},
               appendmat=False, oned_as='row')
    shutil.copyfile(op.join(base_dir, 'test_raw.fdt'),
                    negative_latency_fname.replace('.set', '.fdt'))
    event_id = {eeg.event[0].type: 1}
    pytest.raises(ValueError, read_raw_eeglab, montage=montage, preload=True,
                  event_id=event_id, input_fname=negative_latency_fname)

    # test overlapping events
    overlap_fname = op.join(tmpdir, 'test_overlap_event.set')
    io.savemat(overlap_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': 'test_overlap_event.fdt',
                'epoch': eeg.epoch, 'event': [eeg.event[0], eeg.event[0]],
                'chanlocs': eeg.chanlocs, 'pnts': eeg.pnts}},
               appendmat=False, oned_as='row')
    shutil.copyfile(op.join(base_dir, 'test_raw.fdt'),
                    overlap_fname.replace('.set', '.fdt'))
    event_id = {'rt': 1, 'square': 2}
    with pytest.warns(RuntimeWarning, match='will be dropped'):
        raw = read_raw_eeglab(input_fname=overlap_fname,
                              montage=montage, event_id=event_id,
                              preload=True)
    events_stimchan = find_events(raw)
    events_read_events_eeglab = read_events_eeglab(overlap_fname, event_id)
    assert (len(events_stimchan) == 1)
    assert (len(events_read_events_eeglab) == 2)

    # test reading file with one channel
    one_chan_fname = op.join(tmpdir, 'test_one_channel.set')
    io.savemat(one_chan_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': 1, 'data': np.random.random((1, 3)),
                'epoch': eeg.epoch, 'event': eeg.epoch,
                'chanlocs': {'labels': 'E1', 'Y': -6.6069,
                             'X': 6.3023, 'Z': -2.9423},
                'times': eeg.times[:3], 'pnts': 3}},
               appendmat=False, oned_as='row')
    with pytest.warns(None) as w:
        read_raw_eeglab(input_fname=one_chan_fname, preload=True)
    # no warning for 'no events found'
    assert len(w) == 0

    # test reading file with 3 channels - one without position information
    # first, create chanlocs structured array
    ch_names = ['F3', 'unknown', 'FPz']
    x, y, z = [1., 2., np.nan], [4., 5., np.nan], [7., 8., np.nan]
    dt = [('labels', 'S10'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')]
    chanlocs = np.zeros((3,), dtype=dt)
    for ind, vals in enumerate(zip(ch_names, x, y, z)):
        for fld in range(4):
            chanlocs[ind][dt[fld][0]] = vals[fld]

    if LooseVersion(np.__version__) == '1.14.0':
        # There is a bug in 1.14.0 (or maybe with SciPy 1.0.0?) that causes
        # this write to fail!
        raise SkipTest('Need to fix bug in NumPy 1.14.0!')

    # save set file
    one_chanpos_fname = op.join(tmpdir, 'test_chanpos.set')
    io.savemat(one_chanpos_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': 3, 'data': np.random.random((3, 3)),
                'epoch': eeg.epoch, 'event': eeg.epoch,
                'chanlocs': chanlocs, 'times': eeg.times[:3], 'pnts': 3}},
               appendmat=False, oned_as='row')
    # load it
    with pytest.warns(RuntimeWarning, match='did not have a position'):
        raw = read_raw_eeglab(input_fname=one_chanpos_fname, preload=True)
    # position should be present for first two channels
    for i in range(2):
        assert_array_equal(raw.info['chs'][i]['loc'][:3],
                           np.array([-chanlocs[i]['Y'],
                                     chanlocs[i]['X'],
                                     chanlocs[i]['Z']]))
    # position of the last channel should be zero
    assert_array_equal(raw.info['chs'][-1]['loc'][:3], [np.nan] * 3)

    # test reading channel names from set and positions from montage
    with pytest.warns(RuntimeWarning, match='did not have a position'):
        raw = read_raw_eeglab(input_fname=one_chanpos_fname, preload=True,
                              montage=montage)

    # when montage was passed - channel positions should be taken from there
    correct_pos = [[-0.56705965, 0.67706631, 0.46906776], [np.nan] * 3,
                   [0., 0.99977915, -0.02101571]]
    for ch_ind in range(3):
        assert_array_almost_equal(raw.info['chs'][ch_ind]['loc'][:3],
                                  np.array(correct_pos[ch_ind]))

    # test reading channel names but not positions when there is no X (only Z)
    # field in the EEG.chanlocs structure
    nopos_chanlocs = chanlocs[['labels', 'Z']]
    nopos_fname = op.join(tmpdir, 'test_no_chanpos.set')
    io.savemat(nopos_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate, 'nbchan': 3,
                'data': np.random.random((3, 2)), 'epoch': eeg.epoch,
                'event': eeg.epoch, 'chanlocs': nopos_chanlocs,
                'times': eeg.times[:2], 'pnts': 2}},
               appendmat=False, oned_as='row')
    # load the file
    raw = read_raw_eeglab(input_fname=nopos_fname, preload=True)
    # test that channel names have been loaded but not channel positions
    for i in range(3):
        assert_equal(raw.info['chs'][i]['ch_name'], ch_names[i])
        assert_array_equal(raw.info['chs'][i]['loc'][:3],
                           np.array([np.nan, np.nan, np.nan]))


@requires_h5py
@testing.requires_testing_data
@pytest.mark.parametrize('fnames', [epochs_mat_fnames, epochs_h5_fnames])
def test_io_set_epochs(fnames):
    """Test importing EEGLAB .set epochs files."""
    epochs_fname, epochs_fname_onefile = fnames
    with pytest.warns(RuntimeWarning, match='multiple events'):
        epochs = read_epochs_eeglab(epochs_fname)
    with pytest.warns(RuntimeWarning, match='multiple events'):
        epochs2 = read_epochs_eeglab(epochs_fname_onefile)
    # one warning for each read_epochs_eeglab because both files have epochs
    # associated with multiple events
    assert_array_equal(epochs.get_data(), epochs2.get_data())


@testing.requires_testing_data
def test_io_set_epochs_events(tmpdir):
    """Test different combinations of events and event_ids."""
    tmpdir = str(tmpdir)
    out_fname = op.join(tmpdir, 'test-eve.fif')
    events = np.array([[4, 0, 1], [12, 0, 2], [20, 0, 3], [26, 0,  3]])
    write_events(out_fname, events)
    event_id = {'S255/S8': 1, 'S8': 2, 'S255/S9': 3}
    out_fname = op.join(tmpdir, 'test-eve.fif')
    epochs = read_epochs_eeglab(epochs_fname_mat, events, event_id)
    assert_equal(len(epochs.events), 4)
    assert epochs.preload
    assert epochs._bad_dropped
    epochs = read_epochs_eeglab(epochs_fname_mat, out_fname, event_id)
    pytest.raises(ValueError, read_epochs_eeglab, epochs_fname_mat,
                  None, event_id)
    pytest.raises(ValueError, read_epochs_eeglab, epochs_fname_mat,
                  epochs.events, None)


@testing.requires_testing_data
def test_degenerate(tmpdir):
    """Test some degenerate conditions."""
    # test if .dat file raises an error
    tmpdir = str(tmpdir)
    eeg = io.loadmat(epochs_fname_mat, struct_as_record=False,
                     squeeze_me=True)['EEG']
    eeg.data = 'epochs_fname.dat'
    bad_epochs_fname = op.join(tmpdir, 'test_epochs.set')
    io.savemat(bad_epochs_fname, {'EEG':
               {'trials': eeg.trials, 'srate': eeg.srate,
                'nbchan': eeg.nbchan, 'data': eeg.data,
                'epoch': eeg.epoch, 'event': eeg.event,
                'chanlocs': eeg.chanlocs, 'pnts': eeg.pnts}},
               appendmat=False, oned_as='row')
    shutil.copyfile(op.join(base_dir, 'test_epochs.fdt'),
                    op.join(tmpdir, 'test_epochs.dat'))
    with pytest.warns(RuntimeWarning, match='multiple events'):
        pytest.raises(NotImplementedError, read_epochs_eeglab,
                      bad_epochs_fname)


@pytest.mark.parametrize("fname", raw_fnames)
@testing.requires_testing_data
def test_eeglab_annotations(fname):
    """Test reading annotations in EEGLAB files."""
    _check_h5(fname)
    annotations = read_annotations_eeglab(fname)
    assert len(annotations) == 154
    assert set(annotations.description) == set(['rt', 'square'])
    assert np.all(annotations.duration == 0.)


run_tests_if_main()
