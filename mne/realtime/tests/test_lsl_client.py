# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
from random import random as rand
import numpy as np

from mne.realtime import LSLClient, MockLSLStream, RtEpochs
from mne.utils import run_tests_if_main, requires_pylsl
from mne import create_info
from mne.io import constants


host = 'myuid34234'
event_id = 5
n_channels = 8
sfreq = 100
tmin = -0.1
tmax = 0.5
ch_types = ["eeg" for n in range(n_channels - 1)] + ['stim']
ch_names = ["MNE {:03d}".format(ch_id) for ch_id
            in range(1, n_channels + 1)]
stim_channel = ch_names[-1]

info = create_info(ch_names, sfreq, ch_types)

@requires_pylsl
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    n_channels = 8
    n_samples = 5
    wait_max = 10

    stream = MockLSLStream(host, n_channels, testing=True)
    stream.start()

    with LSLClient(info=info, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()
        epoch = client.get_data_as_epoch(n_samples=n_samples)

    assert client_info['nchan'] == n_channels
    assert ([ch["ch_name"] for ch in client_info["chs"]] ==
            [ch_name for ch_name in ch_names])
    assert any([constants.FIFF.FIFFV_STIM_CH == ch['kind']
                for ch in info['chs']])
    assert n_channels, n_samples == epoch.get_data().shape[1:]

    stream.stop()


run_tests_if_main()
