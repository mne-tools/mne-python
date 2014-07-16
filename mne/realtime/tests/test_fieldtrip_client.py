# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import time
import os
import threading
import subprocess
import warnings
import os.path as op

from nose.tools import assert_true

from mne.utils import requires_neuromag2ft
from mne.realtime import FieldTripClient
from mne.externals.six.moves import queue

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.realpath(op.join(base_dir, 'test_raw.fif'))

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _run_buffer(kill_signal, neuromag2ft_fname):
    cmd = (neuromag2ft_fname, '--file', raw_fname, '--speed', '4.0')

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    # Let measurement continue for the entire duration
    kill_signal.get(timeout=10.0)
    print('Terminating subprocess')
    process.terminate()


@requires_neuromag2ft
def test_fieldtrip_client():
    """Test fieldtrip_client"""

    neuromag2ft_fname = op.realpath(op.join(os.environ['NEUROMAG2FT_ROOT'],
                                    'neuromag2ft'))

    kill_signal = queue.Queue()
    thread = threading.Thread(target=_run_buffer, args=(kill_signal,
                                                        neuromag2ft_fname))
    thread.daemon = True
    thread.start()

    # Start the FieldTrip buffer
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FieldTripClient(host='localhost', port=1972,
                             tmax=5, wait_max=1) as rt_client:
            tmin_samp1 = rt_client.tmin_samp

    time.sleep(1)  # Pause measurement
    assert_true(len(w) == 1)

    # Start the FieldTrip buffer again
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with FieldTripClient(host='localhost', port=1972,
                             tmax=5, wait_max=1) as rt_client:
            print(rt_client.tmin_samp)
            tmin_samp2 = rt_client.tmin_samp

    kill_signal.put(False)  # stop the buffer
    assert_true(tmin_samp2 > tmin_samp1)
    assert_true(len(w) == 1)
