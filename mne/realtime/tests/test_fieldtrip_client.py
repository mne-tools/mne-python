# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import time
import os
import threading
import subprocess
import warnings
import os.path as op

from nose.tools import assert_true, assert_equal

import mne
from mne.utils import requires_neuromag2ft, run_tests_if_main
from mne.realtime import FieldTripClient
from mne.externals.six.moves import queue

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.realpath(op.join(base_dir, 'test_raw.fif'))

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _run_buffer(kill_signal, neuromag2ft_fname):
    # Works with neuromag2ft-3.0.2
    cmd = (neuromag2ft_fname, '--file', raw_fname, '--speed', '4.0')

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    # Let measurement continue for the entire duration
    kill_signal.get(timeout=10.0)
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
    time.sleep(0.25)

    try:
        # Start the FieldTrip buffer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with FieldTripClient(host='localhost', port=1972,
                                 tmax=5, wait_max=1) as rt_client:
                tmin_samp1 = rt_client.tmin_samp

        time.sleep(1)  # Pause measurement
        assert_true(len(w) >= 1)

        # Start the FieldTrip buffer again
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with FieldTripClient(host='localhost', port=1972,
                                 tmax=5, wait_max=1) as rt_client:
                raw_info = rt_client.get_measurement_info()

                tmin_samp2 = rt_client.tmin_samp
                picks = mne.pick_types(raw_info, meg='grad', eeg=False,
                                       stim=False, eog=False)
                epoch = rt_client.get_data_as_epoch(n_samples=5, picks=picks)
                n_channels, n_samples = epoch.get_data().shape[1:]

                epoch2 = rt_client.get_data_as_epoch(n_samples=5, picks=picks)
                n_channels2, n_samples2 = epoch2.get_data().shape[1:]

                # case of picks=None
                epoch = rt_client.get_data_as_epoch(n_samples=5)

        assert_true(tmin_samp2 > tmin_samp1)
        assert_true(len(w) >= 1)
        assert_equal(n_samples, 5)
        assert_equal(n_samples2, 5)
        assert_equal(n_channels, len(picks))
        assert_equal(n_channels2, len(picks))
        kill_signal.put(False)  # stop the buffer
    except:
        kill_signal.put(False)  # stop the buffer even if tests fail
        raise


run_tests_if_main()
