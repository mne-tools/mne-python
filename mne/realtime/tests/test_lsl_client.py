# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
from os import getenv, path as op
import pytest

from mne.realtime import LSLClient, MockLSLStream
from mne.utils import run_tests_if_main, requires_pylsl
from mne.io import read_raw_fif
from mne.datasets import testing


host = 'myuid34234'
base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')


@requires_pylsl
@testing.requires_testing_data
@pytest.mark.skipif(getenv('AZURE_CI_WINDOWS', 'false').lower() == 'true',
                    reason=('Running multiprocessing on Windows ' +
                            'creates a BrokenPipeError, see ' +
                            'https://stackoverflow.com/questions/50079165/'))
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    wait_max = 10

    raw = read_raw_fif(raw_fname)
    n_secs = 1
    raw.crop(n_secs)
    raw_info = raw.info
    sfreq = raw_info['sfreq']
    stream = MockLSLStream(host, raw, ch_type='eeg')
    stream.start()

    with LSLClient(info=raw_info, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()
        epoch = client.get_data_as_epoch(n_samples=sfreq * n_secs * 2)

    assert client_info['nchan'] == raw_info['nchan']
    assert ([ch["ch_name"] for ch in client_info["chs"]] ==
            [ch_name for ch_name in raw_info['ch_names']])

    assert raw_info['nchan'], sfreq == epoch.get_data().shape[1:]

    stream.stop()


run_tests_if_main()
