"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet


n_channels = 8
info = StreamInfo('MNE', 'EEG', n_channels, 100, 'float32', 'myuid34234')
info.desc().append_child_value("manufacturer", "MNE")
channels = info.desc().append_child("channels")
for c_id in range(1, n_channels + 1):
    channels.append_child("channel") \
        .append_child_value("label", "MNE {:03d}".format(c_id)) \
        .append_child_value("type", "eeg") \
        .append_child_value("unit", "microvolts")

# next make an outlet
outlet = StreamOutlet(info)

print("now sending data...")
while True:
    # make a new random 8-channel sample; this is converted into a
    # pylsl.vectorf (the data type that is expected by push_sample)
    mysample = [rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand()]
    mysample = [x*1e-6 for x in mysample]
    # now send it and wait for a bit
    outlet.push_sample(mysample)
    time.sleep(0.01)
