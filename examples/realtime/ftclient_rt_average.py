"""
========================================================
Compute real-time evoked responses with FieldTrip client
========================================================

This example demonstrates how to connect the MNE real-time
system to the Fieldtrip buffer using FieldTripClient class.

This example was tested in simulation mode

neuromag2ft --file MNE-sample-data/MEG/sample/sample_audvis_raw.fif

using a modified version of neuromag2ft available at

http://neuro.hut.fi/~mainak/neuromag2ft-2.0.0.zip

to run the FieldTrip buffer. Then running this example acquires the
data on the client side.

Since the Fieldtrip buffer does not contain all the
measurement information required by the MNE real-time processing
pipeline, an info dictionary must be provided to instantiate FieldTripClient.
Alternatively, the MNE-Python script will try to guess the missing
measurement info from the Fieldtrip Header object.

Together with RtEpochs, this can be used to compute evoked
responses using moving averages.
"""

print(__doc__)

# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import mne
from mne.realtime import FieldTripClient, RtEpochs

import matplotlib.pyplot as plt

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
bads = ['MEG 2443', 'EEG 053']

plt.ion()  # make plot interactive

# 'with' statement is required for a clean exit
with FieldTripClient(host='localhost', port=1972,
                     tmax=150) as rt_client:

    # get measurement info guessed by MNE-Python
    raw_info = rt_client.get_measurement_info()

    # select gradiometers
    picks = mne.pick_types(raw_info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=bads)

    # create the real-time epochs object
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         reject=dict(grad=4000, eog=150),
                         decim=1, isi_max=10.0, proj=None)

    # start the acquisition
    rt_epochs.start()

    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        print("Just got epoch %d" % (ii + 1))
        if ii > 0:
            ev += evoked
        evoked = ev
        plt.clf()  # clear canvas
        evoked.plot(axes=plt.gca())  # plot on current figure
        plt.pause(0.05)

    plt.close()
