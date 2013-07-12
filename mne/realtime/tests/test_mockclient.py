import os.path as op

import mne
from mne.realtime import MockRtClient, RtEpochs

from numpy.testing import assert_array_almost_equal

base_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

raw = mne.fiff.Raw(raw_fname, preload=True, verbose=False)

picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])


def test_mockclient():
    """Test whether the data sent by mockclient is the same as in the raw file
    """

    n_epochs, event_id, tmin, tmax = 1, 1, -0.2, 0.5

    rt_client = MockRtClient(raw)
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, n_epochs,
                         consume_epochs=False, picks=picks)

    rt_epochs.start()
    rt_client.send_data(rt_epochs, tmin=0, tmax=7, buffer_size=1000)

    sfreq = raw.info['sfreq']
    event_samp = rt_epochs.events[0][0]

    tmin_samp = int(round(sfreq * tmin)) + event_samp
    tmax_samp = tmin_samp + len(rt_epochs._raw_times)

    data_raw, _ = raw[picks, tmin_samp:tmax_samp]
    data_proc = rt_epochs._epoch_queue[0]

    assert_array_almost_equal(data_raw, data_proc, decimal=10)
