import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_equal

import mne
from mne import Epochs, read_events
from mne.realtime import MockRtClient, RtEpochs

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

raw = mne.io.Raw(raw_fname, preload=True, verbose=False)

events = read_events(event_name)

picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=True, exclude=raw.info['bads'])


def test_mockclient():
    """Test the RtMockClient."""

    event_id, tmin, tmax = 1, -0.2, 0.5

    epochs = Epochs(raw, events[:7], event_id=event_id, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(None, 0), preload=True)
    data = epochs.get_data()

    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)

    rt_data = rt_epochs.get_data()

    assert_true(rt_data.shape == data.shape)
    assert_array_equal(rt_data, data)


def test_get_event_data():
    """Test emulation of realtime data stream."""

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
