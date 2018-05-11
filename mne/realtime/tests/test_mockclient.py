import os.path as op
import time

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mne
from mne import Epochs, read_events, pick_channels
from mne.utils import (run_tests_if_main, _TempDir)
from mne.realtime import MockRtClient, RtEpochs

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

events = read_events(event_name)


def test_mockclient():
    """Test the RtMockClient."""

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=raw.info['bads'])

    event_id, tmin, tmax = 1, -0.2, 0.5

    epochs = Epochs(raw, events[:7], event_id=event_id, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(None, 0), preload=True)
    data = epochs.get_data()

    rt_client = MockRtClient(raw)
    # choose "large" value, should always be longer than execution time of
    # get_data()
    isi_max = 0.5
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         isi_max=isi_max)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)

    # get_data() should return immediately and not wait for the timeout
    start_time = time.time()
    rt_data = rt_epochs.get_data()
    retrieval_time = time.time() - start_time
    assert retrieval_time < isi_max
    assert rt_data.shape == data.shape
    assert_array_equal(rt_data, data)
    assert len(rt_epochs) == len(epochs)

    # iteration over epochs should block until timeout
    rt_iter_data = list()
    start_time = time.time()
    for cur_epoch in rt_epochs:
        rt_iter_data.append(cur_epoch)
    retrieval_time = time.time() - start_time
    assert retrieval_time >= isi_max
    rt_iter_data = np.array(rt_iter_data)
    assert rt_iter_data.shape == data.shape
    assert_array_equal(rt_iter_data, data)
    assert len(rt_epochs) == len(epochs)

    tempdir = _TempDir()  # will be removed when out of scope
    export_file = str(op.join(tempdir, 'test_rt-epo.fif'))
    rt_epochs.save(export_file)

    loaded_epochs = mne.read_epochs(export_file)
    loaded_data = loaded_epochs.get_data()
    assert rt_data.shape == loaded_data.shape
    assert_allclose(loaded_data, rt_data)


def test_get_event_data():
    """Test emulation of realtime data stream."""

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=raw.info['bads'])

    event_id, tmin, tmax = 2, -0.1, 0.3
    epochs = Epochs(raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax, picks=picks, baseline=None,
                    preload=True, proj=False)

    data = epochs.get_data()[0, :, :]

    rt_client = MockRtClient(raw)
    rt_data = rt_client.get_event_data(event_id=event_id, tmin=tmin,
                                       tmax=tmax, picks=picks,
                                       stim_channel='STI 014')

    assert_array_equal(rt_data, data)


def test_find_events():
    """Test find_events in rt_epochs."""

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=raw.info['bads'])

    event_id = [0, 5, 6]
    tmin, tmax = -0.2, 0.5

    stim_channel = 'STI 014'
    stim_channel_idx = pick_channels(raw.info['ch_names'],
                                     include=[stim_channel])

    # Reset some data for ease of comparison
    raw._first_samps[0] = 0
    raw.info['sfreq'] = 1000
    # Test that we can handle consecutive events with no gap
    raw._data[stim_channel_idx, :] = 0
    raw._data[stim_channel_idx, 500:520] = 5
    raw._data[stim_channel_idx, 520:530] = 6
    raw._data[stim_channel_idx, 530:532] = 5
    raw._data[stim_channel_idx, 540] = 6
    raw._update_times()

    # consecutive=False
    find_events = dict(consecutive=False)

    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         stim_channel='STI 014', isi_max=0.5,
                         find_events=find_events)
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)
    rt_epochs.start()
    events = [5, 6]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert_true(ev.comment == str(events[ii]))
    assert_true(ii == 1)

    # consecutive=True
    find_events = dict(consecutive=True)
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         stim_channel='STI 014', isi_max=0.5,
                         find_events=find_events)
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)
    rt_epochs.start()
    events = [5, 6, 5, 6]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert_true(ev.comment == str(events[ii]))
    assert_true(ii == 3)

    # min_duration=0.002
    find_events = dict(consecutive=False, min_duration=0.002)
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         stim_channel='STI 014', isi_max=0.5,
                         find_events=find_events)
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)
    rt_epochs.start()
    events = [5]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert_true(ev.comment == str(events[ii]))
    assert_true(ii == 0)

    # output='step', consecutive=True
    find_events = dict(output='step', consecutive=True)
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         stim_channel='STI 014', isi_max=0.5,
                         find_events=find_events)
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)
    rt_epochs.start()
    events = [5, 6, 5, 0, 6, 0]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert_true(ev.comment == str(events[ii]))
    assert_true(ii == 5)

    # Reset some data for ease of comparison
    raw._first_samps[0] = 0
    raw.info['sfreq'] = 1000
    # Test that we can handle events at the beginning of the buffer
    raw._data[stim_channel_idx, :] = 0
    raw._data[stim_channel_idx, 1000:1005] = 5
    raw._update_times()

    # Check that we find events that start at the beginning of the buffer
    find_events = dict(consecutive=False)
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         stim_channel='STI 014', isi_max=0.5,
                         find_events=find_events)
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)
    rt_epochs.start()
    events = [5]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert_true(ev.comment == str(events[ii]))
    assert_true(ii == 0)

    # Reset some data for ease of comparison
    raw._first_samps[0] = 0
    raw.info['sfreq'] = 1000
    # Test that we can handle events over different buffers
    raw._data[stim_channel_idx, :] = 0
    raw._data[stim_channel_idx, 997:1003] = 5
    raw._update_times()
    for min_dur in [0.002, 0.004]:
        find_events = dict(consecutive=False, min_duration=min_dur)
        rt_client = MockRtClient(raw)
        rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                             stim_channel='STI 014', isi_max=0.5,
                             find_events=find_events)
        rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10,
                            buffer_size=1000)
        rt_epochs.start()
        events = [5]
        for ii, ev in enumerate(rt_epochs.iter_evoked()):
            assert_true(ev.comment == str(events[ii]))
        assert_true(ii == 0)


def test_rejection():
    event_id, tmin, tmax = 1, 0.0, 0.5
    sfreq = 1000
    ch_names = ['Fz', 'Cz', 'Pz', 'STI 014']
    raw_tmax = 5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=['eeg', 'eeg', 'eeg', 'stim'])
    raw_array = np.random.randn(len(ch_names), raw_tmax * sfreq)
    raw_array[-1, :] = 0
    epoch_start_samples = np.arange(raw_tmax) * sfreq
    raw_array[-1, epoch_start_samples] = event_id

    reject_threshold = np.max(raw_array) - np.min(raw_array) + 1
    reject = {'eeg': reject_threshold}
    epochs_to_reject = [1, 3]
    epochs_to_keep = np.setdiff1d(np.arange(len(epoch_start_samples)),
                                  epochs_to_reject)
    expected_drop_log = [list() for _ in range(len(epoch_start_samples))]
    for cur_epoch in epochs_to_reject:
        raw_array[1, epoch_start_samples[cur_epoch]] = reject_threshold + 1
        expected_drop_log[cur_epoch] = [ch_names[1]]

    raw = mne.io.RawArray(raw_array, info)
    events = mne.find_events(raw, shortest_event=1, initial_event=True)
    picks = mne.pick_types(raw.info, eeg=True)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, picks=picks, preload=True,
                        reject=reject)
    epochs_data = epochs.get_data()

    assert len(epochs) == len(epoch_start_samples) - len(epochs_to_reject)
    assert_array_equal(epochs_data[:, 1, 0],
                       raw_array[1, epoch_start_samples[epochs_to_keep]])
    assert_array_equal(epochs.drop_log, expected_drop_log)
    assert_array_equal(epochs.selection, epochs_to_keep)

    rt_client = MockRtClient(raw)

    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         baseline=None, isi_max=0.5,
                         find_events=dict(initial_event=True),
                         reject=reject)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=raw_tmax,
                        buffer_size=250)

    assert len(rt_epochs) == len(epochs_to_keep)
    assert_array_equal(rt_epochs.drop_log, expected_drop_log)
    assert_array_equal(rt_epochs.selection, epochs_to_keep)
    rt_data = rt_epochs.get_data()
    assert rt_data.shape == epochs_data.shape
    assert_array_equal(rt_data, epochs_data)


run_tests_if_main()
