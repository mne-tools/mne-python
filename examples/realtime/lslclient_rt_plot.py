"""
==============================================================
Plot real-time epoch data with LSL client
==============================================================

This example demonstrates how to use the LSL client to plot real-time
collection of event data from an LSL stream.
For the purposes of demo, a mock LSL stream is constructed. You can
replace this with the stream of your choice by changing the host id to
the desired stream.

"""
# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)
from multiprocessing import Process
import time
from random import random as rand

import numpy as np
import matplotlib.pyplot as plt

from mne.realtime import LSLClient

print(__doc__)

# this is the host id that identifies your stream on LSL
host = 'mne_stream'
# this is the max wait time in seconds until client connection
wait_max = 5


# this is a helper function that will simulate an LSL stream
def _start_mock_lsl_stream(host):
    """Start a mock LSL stream to test LSLClient."""
    from pylsl import StreamInfo, StreamOutlet

    n_channels = 8
    sfreq = 1000
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
    rands = [rand(), rand(), rand(), rand(),
             rand(), rand(), rand(), rand()]
    print("now sending data...")
    counter = 0
    while True:
        sample = counter % 40
        const = np.sin(2 * np.pi * sample / 40)
        mysample = [x * const * 1e-6 for x in rands]
        # now send it and wait for a bit
        outlet.push_sample(mysample)
        counter += 1
        time.sleep(sfreq**-1)


# Let's start our mock LSL stream here
process = Process(target=_start_mock_lsl_stream, args=(host,))
process.daemon = True
process.start()  # Now there should be streaming data be generated

# Let's observe it
plt.ion()  # make plot interactive
_, ax = plt.subplots(1)
with LSLClient(info=None, host=host, wait_max=wait_max) as client:
    client_info = client.get_measurement_info()
    print(client_info)

    # let's observe ten seconds of data
    for ii in range(10):
        epoch = client.get_data_as_epoch(n_samples=100)
        epoch.average().plot(axes=ax, ylim=dict(eeg=[-1, 1]))
        plt.pause(1)
        plt.cla()

# Let's terminate the mock LSL stream
process.terminate()
plt.close()
