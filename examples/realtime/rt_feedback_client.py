"""
==============================================
Real-time feedback for decoding :: Client Side
==============================================

This example demonstrates how to setup a real-time feedback
mechanism using StimServer and StimClient.

The idea here is to display future stimuli for the class which
is predicted less accurately. This allows on-demand adaptation
of the stimuli depending on the needs of the classifier.

To run this example, open ipython in two separate terminals.
In the first, run rt_feedback_server.py and then wait for the
message

    RtServer: Start

Once that appears, run rt_feedback_client.py in the other terminal
and the feedback script should start.

All brain responses are simulated from a fiff file to make it easy
to test. However, it should be possible to adapt this script
for a real experiment.

"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from mne.realtime import StimClient
import time

print(__doc__)

# Instantiating stimulation client

# Port number must match port number used to instantiate
# StimServer. Any port number above 1000 should be fine
# because they do not require root permission.
stim_client = StimClient('localhost', port=4218)


ev_list = list()  # list of events displayed

# start with right checkerboard stimuli. This is required
# because the ev_list.append(ev_list[-1]) will not work
# if ev_list is empty.
trig = 4
stim_duration = 1.0

# iterating over 50 epochs
for ii in range(50):

    if trig is not None:
        ev_list.append(trig)  # use the last trigger received
    else:
        ev_list.append(ev_list[-1])  # use the last stimuli

    # draw left or right checkerboard according to ev_list
    if ev_list[ii] == 3:
        print('Stimulus: left checkerboard')
    else:
        print('Stimulus: right checkerboard')

    last_stim_time = time.time()
    trig = stim_client.get_trigger(timeout=(stim_duration - 0.05))

    time.sleep(max(stim_duration - (time.time() - last_stim_time), 0))

    print('Stimulus: Fixation Cross')
