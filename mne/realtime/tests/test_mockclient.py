import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_equal

import mne
from mne import Epochs, read_events, pick_channels
from mne.utils import run_tests_if_main
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
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         isi_max=0.5)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)

    rt_data = rt_epochs.get_data()

    assert_true(rt_data.shape == data.shape)
    assert_array_equal(rt_data, data)


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


run_tests_if_main()
