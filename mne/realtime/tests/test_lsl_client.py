# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import subprocess
import os
import os.path as op
import time
from random import random as rand
import signal

import numpy as np
import pytest

from mne.realtime import LSLClient
from mne.utils import run_tests_if_main


def _start_fake_lsl_stream():
    """Start a fake LSL stream to test LSLClient."""
    stream_file = op.join(op.dirname(__file__), '_start_fake_lsl_stream.py')
    cmd = ('python', stream_file)

    # sleep to make sure everything is setup before playback starts
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    return process


def _stop_fake_lsl_stream(process):
    """Terminate a fake LSL stream subprocess."""
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    return self


def test_lsl_client():
    """Test the LSLClient for connection and data retrieval."""
    process = _run_fake_lsl_stream()
    identifier = 'myuid34234'
    n_chan = 8

    with LSLClient(identifier) as client:
        client_info = client.get_measurement_info()
        assert (ch["ch_name"] for ch in client_info["chs"] ==
                "MNE {:03d}".format(ch_id) for ch_id in range(1, n_chan + 1))

        epoch = client.get_data_as_epoch(n_samples=5)
        assert (n_channels, n_samples == epoch.get_data().shape[1:])

    _terminate_fake_lsl_stream(process)

run_tests_if_main()
