"""
========================================================
Compute real-time evoked responses with FieldTrip client
========================================================

This example demonstrates how to connect the MNE real-time
system to the Fieldtrip buffer using FieldTripClient class.

Since the Fieldtrip buffer does not contain all the
measurement information required by the MNE real-time processing
pipeline, a raw object must be provided to instantiate MneFtClient.
Together with RtEpochs, this can be used to compute evoked
responses using moving averages.
"""

print(__doc__)

# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.realtime import FieldTripClient, RtEpochs

import matplotlib.pyplot as plt

# file containing measurement information
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.fiff.Raw(raw_fname, preload=False)

# select gradiometers
picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# 'with' statement is required for a clean exit
with FieldTripClient(raw=raw, host='localhost', port=1972,
                     tmax=150) as rt_client:

    # create the real-time epochs object
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax, picks=picks,
                         decim=1, reject=dict(grad=4000e-13, eog=150e-6),
                         isi_max=10.0)

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
