# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
from random import random as rand

from mne.realtime import LSLClient, MockLSLStream, RtEpochs
from mne.utils import run_tests_if_main, requires_pylsl
from mne import create_info


host = 'myuid34234'


@requires_pylsl
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    n_channels = 8
    n_samples = 5
    wait_max = 10

    stream = MockLSLStream(host, n_channels, testing=True)
    stream.start()

    with LSLClient(info=None, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()

        assert ([ch["ch_name"] for ch in client_info["chs"]] ==
                ["MNE {:03d}".format(ch_id) for ch_id in
                 range(1, n_channels + 1)])

        epoch = client.get_data_as_epoch(n_samples=n_samples)
        assert n_channels, n_samples == epoch.get_data().shape[1:]

    stream.close()


def test_lsl_rt_epochs():
    """Test the functionality of the LSL Client with RtEpochs."""
    event_id = 5
    n_channels = 8
    sfreq = 100
    tmin = -0.1
    tmax = 0.5
    ch_types = ["eeg" for n in range(n_channels - 1)] + ['stim']
    ch_names = ["MNE {:03d}".format(ch_id) for ch_id
                in range(1, n_channels + 1)]
    stim_channel = ch_names[-1]

    stream = MockLSLStream(host, n_channels=n_channels, ch_type="eeg",
                           sfreq=sfreq, testing=True)
    stream.start()

    info = create_info(ch_names, sfreq, ch_types)
    with LSLClient(info=info, host=host) as client:
        epochs_rt = RtEpochs(client, event_id, tmin, tmax, stim_channel)
        epochs_rt.start()
        time.sleep(10)
        epochs.stop(stop_receive_thread=True)

        data = epochs_rt.get_data()
        n_samples_prestim = np.abs(tmin * sfreq)
        n_sample_poststim = tmax * sfreq

        data_channels, data_samples = data.get_data().shape[1:]
        assert data_channels == n_channels
        assert data_samples == n_samples_prestim + n_sample_poststim

        data_expected = np.ones((n_channels,
                                 n_samples_prestim + n_sample_poststim))
        data_expected[:,:n_samples_prestim] = event_id - 1
        data_expected[:,n_samples_prestim:] = event_id

        assert data == data_expected

run_tests_if_main()
