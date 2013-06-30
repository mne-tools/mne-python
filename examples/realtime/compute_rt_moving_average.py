"""
========================================================
Compute real-time evoked responses using moving averages
========================================================

This example demonstrates how to connect to an MNE Real-time server
using the RtClient and use it together with RtEpochs to compute
evoked responses using moving averages.

Note: The MNE Real-time server (mne_rt_server), which is part of mne-cpp,
has to be running on the same computer.
"""

print __doc__

# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import mne
import pylab as pl

from mne.realtime import RtClient, RtEpochs

# connect to the server and get the measurement info
client = RtClient('localhost')
info = client.get_measurement_info()

# select gradiometers
picks = mne.fiff.pick_types(info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=info['bads'])

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# the number of epochs to average
n_epochs = 100

# create the real-time epochs object
rt_epochs = RtEpochs(client, event_id, tmin, tmax, n_epochs,
                     consume_epochs=False, picks=picks, decim=1,
                     reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# make the plot interactive
pl.ion()

evoked = None

for ev in rt_epochs.iter_evoked():

    if evoked is None:
        evoked = ev
    else:
        evoked += ev

    pl.figure(1)

    evoked.plot()
    pl.show()
    pl.waitforbuttonpress(0.1)
