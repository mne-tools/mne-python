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
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne.viz import plot_events
from mne.realtime import FieldTripClient, RtEpochs

print(__doc__)

# select the left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5

# user must provide list of bad channels because
# FieldTrip header object does not provide that
bads = ['MEG 2443', 'EEG 053']

plt.ion()  # make plot interactive
_, ax = plt.subplots(2, 1, figsize=(8, 8))  # create subplots

with FieldTripClient(host='localhost', port=1972,
                     tmax=150, wait_max=10) as rt_client:

    # get measurement info guessed by MNE-Python
    raw_info = rt_client.get_measurement_info()

    # select gradiometers
    picks = mne.pick_types(raw_info, meg='grad', eeg=False, eog=True,
                           stim=True, exclude=bads)

    # create the real-time epochs object
    rt_epochs = RtEpochs(rt_client, event_id, tmin, tmax,
                         stim_channel='STI 014', picks=picks,
                         reject=dict(grad=4000e-13, eog=150e-6),
                         decim=1, isi_max=10.0, proj=None)

    # start the acquisition
    rt_epochs.start()

    for ii, ev in enumerate(rt_epochs.iter_evoked()):
        print("Just got epoch %d" % (ii + 1))

        ev.pick_types(meg=True, eog=False)
        if ii == 0:
            evoked = ev
        else:
            evoked = mne.combine_evoked([evoked, ev], weights='nave')

        ax[0].cla()
        ax[1].cla()  # clear axis

        plot_events(rt_epochs.events[-5:], sfreq=ev.info['sfreq'],
                    first_samp=-rt_client.tmin_samp, axes=ax[0])

        evoked.plot(axes=ax[1])  # plot on second subplot
        ax[1].set_title('Evoked response for gradiometer channels'
                        '(event_id = %d)' % event_id)

        plt.pause(0.05)
        plt.draw()

    plt.close()
