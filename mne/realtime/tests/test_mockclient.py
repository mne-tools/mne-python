import os.path as op

import mne
from mne import Epochs, read_events
from mne.realtime import MockRtClient, RtEpochs

from nose.tools import assert_true
from numpy.testing import assert_array_equal

base_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

raw = mne.fiff.Raw(raw_fname, preload=True, verbose=False)

events = read_events(event_name)

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])


def test_mockclient():
    """Test the RtMockClient
    """

    n_epochs, event_id, tmin, tmax = 2, 1, -0.2, 0.5

    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, n_epochs,
                         consume_epochs=False, picks=picks)

    epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(None, 0))

    rt_epochs.start()
    rt_client.send_data(rt_epochs, picks, tmin=0, tmax=10, buffer_size=1000)

    for ii in range(n_epochs):
        x_mock = rt_epochs.get_data()[ii][0]
        x_real = epochs.get_data()[ii, :, :]
        assert_true(x_mock.shape == x_real.shape)
        assert_array_equal(x_mock, x_real)
