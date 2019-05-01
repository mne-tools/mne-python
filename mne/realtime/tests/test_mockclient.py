import os.path as op
import time

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest


from mne import (Epochs, read_events, read_epochs, find_events, create_info,
                 pick_channels, pick_types, concatenate_raws)
from mne.io import RawArray, read_raw_fif
from mne.utils import run_tests_if_main
from mne.realtime import MockRtClient, RtEpochs
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

events = read_events(event_name)


def _call_base_epochs_public_api(epochs, tmpdir):
    """Call all public API methods of an (non-empty) epochs object."""
    # make sure saving and loading returns the same data
    orig_data = epochs.get_data()
    export_file = tmpdir.join('test_rt-epo.fif')
    epochs.save(str(export_file), overwrite=True)
    loaded_epochs = read_epochs(str(export_file))
    loaded_data = loaded_epochs.get_data()
    assert orig_data.shape == loaded_data.shape
    assert_allclose(loaded_data, orig_data)

    # decimation
    epochs_copy = epochs.copy()
    epochs_copy.decimate(1)
    assert epochs_copy.get_data().shape == orig_data.shape
    epochs_copy.info['lowpass'] = 10  # avoid warning
    epochs_copy.decimate(10)
    assert np.abs(10.0 - orig_data.shape[2] /
                  epochs_copy.get_data().shape[2]) <= 1

    # check that methods that require preloaded data fail
    with pytest.raises(RuntimeError):
        epochs.crop(tmin=epochs.tmin,
                    tmax=(epochs.tmin + (epochs.tmax - epochs.tmin) / 2))
    with pytest.raises(RuntimeError):
        epochs.drop_channels(epochs.ch_names[0:1])
    with pytest.raises(RuntimeError):
        epochs.resample(epochs.info['sfreq'] / 10)

    # smoke test
    epochs.standard_error()
    avg_evoked = epochs.average()
    epochs.subtract_evoked(avg_evoked)
    epochs.metadata
    epochs.events
    epochs.ch_names
    epochs.tmin
    epochs.tmax
    epochs.filename
    repr(epochs)
    epochs.plot(show=False)
    # save time by not calling all plot functions
    # epochs.plot_psd(show=False)
    # epochs.plot_drop_log(show=False)
    # epochs.plot_topo_image()
    # epochs.plot_psd_topomap()
    # epochs.plot_image()
    epochs.drop_bad()
    epochs_copy.apply_baseline()
    # do not call since we don't want to make assumptions about events
    # epochs_copy.equalize_event_counts(epochs.event_id.keys())
    epochs_copy.drop([0])


def test_mockclient(tmpdir):
    """Test the RtMockClient."""
    raw = read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
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

    _call_base_epochs_public_api(rt_epochs, tmpdir)


def test_get_event_data():
    """Test emulation of realtime data stream."""
    raw = read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
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
    raw = read_raw_fif(raw_fname, preload=True, verbose=False)
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
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
    # make sure next() works even if no iter-method has been called before
    rt_epochs.next()

    events = [5, 6]
    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        assert ev.comment == str(events[ii])
    assert ii == 1

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
        assert ev.comment == str(events[ii])
    assert ii == 3

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
        assert ev.comment == str(events[ii])
    assert ii == 0

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
        assert ev.comment == str(events[ii])
    assert ii == 5

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
        assert ev.comment == str(events[ii])
    assert ii == 0

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
            assert ev.comment == str(events[ii])
        assert ii == 0


@pytest.mark.parametrize("buffer_size", [420, 1000, 6000])
def test_rejection(buffer_size):
    """Test rejection."""
    event_id, tmin, tmax = 1, 0.0, 0.5
    sfreq = 1000
    ch_names = ['Fz', 'Cz', 'Pz', 'STI 014']
    raw_tmax = 5
    info = create_info(ch_names=ch_names, sfreq=sfreq,
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

    raw = RawArray(raw_array, info)
    events = find_events(raw, shortest_event=1, initial_event=True)
    picks = pick_types(raw.info, eeg=True)
    epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
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
                        buffer_size=buffer_size)

    assert len(rt_epochs) == len(epochs_to_keep)
    assert_array_equal(rt_epochs.drop_log, expected_drop_log)
    assert_array_equal(rt_epochs.selection, epochs_to_keep)
    rt_data = rt_epochs.get_data()
    assert rt_data.shape == epochs_data.shape
    assert_array_equal(rt_data, epochs_data)


@testing.requires_testing_data
def test_events_long():
    """Test events."""
    data_path = testing.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_trunc_raw.fif'
    raw = read_raw_fif(raw_fname, preload=True)
    raw_tmin, raw_tmax = 0, 90

    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, vis_l=3)

    # select gradiometers
    picks = pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=True, exclude=raw.info['bads'])

    # load data with usual Epochs for later verification
    raw = concatenate_raws([raw, raw.copy(), raw.copy(), raw.copy(),
                            raw.copy(), raw.copy()])
    assert 110 < raw.times[-1] < 130
    raw_cropped = raw.copy().crop(raw_tmin, raw_tmax)
    events_offline = find_events(raw_cropped)
    epochs_offline = Epochs(raw_cropped, events_offline, event_id=event_id,
                            tmin=tmin, tmax=tmax, picks=picks, decim=1,
                            reject=dict(grad=4000e-13, eog=150e-6),
                            baseline=None)
    epochs_offline.drop_bad()

    # create the mock-client object
    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks, decim=1,
                         reject=dict(grad=4000e-13, eog=150e-6), baseline=None,
                         isi_max=1.)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=raw_tmin, tmax=raw_tmax,
                        buffer_size=1000)

    expected_events = epochs_offline.events.copy()
    expected_events[:, 0] = expected_events[:, 0] - raw_cropped.first_samp
    assert np.all(expected_events[:, 0] <=
                  (raw_tmax - tmax) * raw.info['sfreq'])
    assert_array_equal(rt_epochs.events, expected_events)
    assert len(rt_epochs) == len(epochs_offline)

    data_picks = pick_types(epochs_offline.info, meg='grad', eeg=False,
                            eog=True,
                            stim=False, exclude=raw.info['bads'])

    for ev_num, ev in enumerate(rt_epochs.iter_evoked()):
        if ev_num == 0:
            X_rt = ev.data[None, data_picks, :]
            y_rt = int(ev.comment)  # comment attribute contains the event_id
        else:
            X_rt = np.concatenate((X_rt, ev.data[None, data_picks, :]), axis=0)
            y_rt = np.append(y_rt, int(ev.comment))

    X_offline = epochs_offline.get_data()[:, data_picks, :]
    y_offline = epochs_offline.events[:, 2]
    assert_array_equal(X_rt, X_offline)
    assert_array_equal(y_rt, y_offline)


run_tests_if_main()
