# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import threading
import time
from random import random as rand

from mne.realtime import LSLClient
from mne.utils import run_tests_if_main, requires_pylsl


host = 'myuid34234'


def _start_mock_lsl_stream(host):
    """Start a mock LSL stream to test LSLClient."""
    from pylsl import StreamInfo, StreamOutlet

    n_channels = 8
    sfreq = 100
    info = StreamInfo('MNE', 'EEG', n_channels, sfreq, 'float32', host)
    info.desc().append_child_value("manufacturer", "MNE")
    channels = info.desc().append_child("channels")
    for c_id in range(1, n_channels + 1):
        channels.append_child("channel") \
                .append_child_value("label", "MNE {:03d}".format(c_id)) \
                .append_child_value("type", "eeg") \
                .append_child_value("unit", "microvolts")

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    while True:
        mysample = [rand(), rand(), rand(), rand(),
                    rand(), rand(), rand(), rand()]
        mysample = [x * 1e-6 for x in mysample]
        # now send it and wait for a bit
        outlet.push_sample(mysample)
        time.sleep(0.01)


@requires_pylsl
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    n_chan = 8
    n_samples = 5
    wait_max = 10

    thread = threading.Thread(target=_start_mock_lsl_stream,
                              args=(host,))
    thread.daemon = True
    thread.start()

    with LSLClient(info=None, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()

        assert ([ch["ch_name"] for ch in client_info["chs"]] ==
                ["MNE {:03d}".format(ch_id) for ch_id in range(1, n_chan + 1)])

        epoch = client.get_data_as_epoch(n_samples=n_samples)
        assert n_chan, n_samples == epoch.get_data().shape[1:]


run_tests_if_main()
