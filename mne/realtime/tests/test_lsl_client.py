# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import os.path as op

from mne.realtime import LSLClient, MockLSLStream
from mne.utils import run_tests_if_main, requires_pylsl
from mne import create_info
from mne.io import constants, read_raw_fif
from mne.datasets import testing

host = 'myuid34234'
base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')

@requires_pylsl
@testing.requires_testing_data
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    wait_max = 10

    raw = read_raw_fif(raw_fname)
    raw_info = raw.info
    stream = MockLSLStream(host, raw, ch_type='eeg')
    stream.start()

    with LSLClient(info=raw_info, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()
        epoch = client.get_data_as_epoch(n_samples=n_samples)

    assert client_info['nchan'] == raw_info['nchan']
    assert ([ch["ch_name"] for ch in client_info["chs"]] ==
            [ch_name for ch_name in raw_info['ch_names']])

    assert raw_info['nchan'], n_samples == epoch.get_data().shape[1:]

    stream.stop()


run_tests_if_main()
