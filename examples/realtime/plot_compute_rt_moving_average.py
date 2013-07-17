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
from mne.datasets import sample
from mne.realtime import RtEpochs, MockRtClient

import pylab as pl

# Fiff file to simulate the realtime client
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.fiff.Raw(raw_fname, preload=True)

# select gradiometers
picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# the number of epochs to average
n_epochs = 30

# create the mock-client object
rt_client = MockRtClient(raw)

# create the real-time epochs object
rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                     decim=1, reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# send raw buffers
rt_client.send_data(rt_epochs, picks, tmin=0, tmax=150, buffer_size=1000)

# make the plot interactive
pl.ion()

evoked = None

for ii, ev in enumerate(rt_epochs.iter_evoked()):

    print "Just got epoch %d" % (ii + 1)

    if evoked is None:
        evoked = ev
    else:
        evoked += ev

    pl.figure(1)

    evoked.plot()
    pl.show()
    pl.waitforbuttonpress(0.1)
