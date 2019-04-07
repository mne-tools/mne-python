# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
import subprocess
import os.path as op

from mne.realtime import LSLClient
from mne.utils import run_tests_if_main, requires_pylsl


host = 'myuid34234'

@requires_pylsl
def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    n_chan = 8
    n_samples = 5
    wait_max = 10

    with LSLClient(info=None, host=host, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()

        assert ([ch["ch_name"] for ch in client_info["chs"]] ==
                ["MNE {:03d}".format(ch_id) for ch_id in range(1, n_chan + 1)])

        epoch = client.get_data_as_epoch(n_samples=n_samples)
        assert n_chan, n_samples == epoch.get_data().shape[1:]


if __name__ == '__main__':
    stream_file = op.join(op.dirname(__file__), '_start_mock_lsl_stream.py')
    cmd = ('python', stream_file)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    run_tests_if_main()
    process.terminate()
