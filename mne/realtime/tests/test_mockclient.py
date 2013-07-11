import os.path as op

import mne
from mne import Epochs, read_events
from mne.realtime import MockRtClient, RtEpochs

from numpy.testing import assert_array_almost_equal

base_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

# Fiff file to simulate the realtime client
raw = mne.fiff.Raw(raw_fname, preload=True, verbose=False)

events = read_events(event_name)

n_epochs, event_id, tmin, tmax = 1, 1, -0.2, 0.5

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                                stim=True, exclude=raw.info['bads'])

#def test_mockclient():

rt_client = MockRtClient(raw)
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, n_epochs,
                     consume_epochs=False, picks=picks)

rt_epochs.start()
rt_client.send_data(rt_epochs, tmin=0, tmax=2, buffer_size=1000)

for ev in rt_epochs.iter_evoked():
    x_mock = ev.data[None, ...]

print "hi"

epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                picks=picks, baseline=(None, 0))
x_real = epochs.get_data()[0,:,:]

#assert_array_almost_equal(x_mock, x_real, decimal=6)
