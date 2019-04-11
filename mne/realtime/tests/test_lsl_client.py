# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import time
from random import random as rand

from mne.realtime import LSLClient, MockLSLStream
from mne.utils import run_tests_if_main, requires_pylsl


host = 'myuid34234'


@requires_pylsl
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    n_channels = 8
    n_samples = 5
    wait_max = 10

    stream = MockLSLStream(host, n_channels)
    process.start()

    with LSLClient(info=None, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()

        assert ([ch["ch_name"] for ch in client_info["chs"]] ==
                ["MNE {:03d}".format(ch_id) for ch_id in
                 range(1, n_channels + 1)])

        epoch = client.get_data_as_epoch(n_samples=n_samples)
        assert n_channels, n_samples == epoch.get_data().shape[1:]

    stream.close()


run_tests_if_main()
