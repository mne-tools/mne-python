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
import matplotlib.pyplot as plt

from mne.realtime import LSLClient, MockLSLStream

print(__doc__)

# this is the host id that identifies your stream on LSL
host = 'mne_stream'
# this is the max wait time in seconds until client connection
wait_max = 5


# For this example, let's use the mock LSL stream.
stream = MockLSLStream(host)
stream.start()

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
stream.close()
plt.close()
