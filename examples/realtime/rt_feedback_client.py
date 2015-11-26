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
from psychopy import visual, core

print(__doc__)

# Instantiating stimulation client

# Port number must match port number used to instantiate
# StimServer. Any port number above 1000 should be fine
# because they do not require root permission.
stim_client = StimClient('localhost', port=4218)

# create a window
mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")

# create the stimuli

# right checkerboard stimuli
right_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                             visibleWedge=[0, 180], radialCycles=4,
                             angularCycles=8, interpolate=False,
                             autoLog=False)

# left checkerboard stimuli
left_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                            visibleWedge=[180, 360], radialCycles=4,
                            angularCycles=8, interpolate=False,
                            autoLog=False)

# fixation dot
fixation = visual.PatchStim(mywin, color=-1, colorSpace='rgb', tex=None,
                            mask='circle', size=0.2)

# the most accurate method is using frame refresh periods
# however, since the actual refresh rate is not known
# we use the Clock
timer1 = core.Clock()
timer2 = core.Clock()

ev_list = list()  # list of events displayed

# start with right checkerboard stimuli. This is required
# because the ev_list.append(ev_list[-1]) will not work
# if ev_list is empty.
trig = 4

# iterating over 50 epochs
for ii in range(50):

    if trig is not None:
        ev_list.append(trig)  # use the last trigger received
    else:
        ev_list.append(ev_list[-1])  # use the last stimuli

    # draw left or right checkerboard according to ev_list
    if ev_list[ii] == 3:
        left_cb.draw()
    else:
        right_cb.draw()

    fixation.draw()  # draw fixation
    mywin.flip()  # show the stimuli

    timer1.reset()  # reset timer
    timer1.add(0.75)  # display stimuli for 0.75 sec

    # return within 0.2 seconds (< 0.75 seconds) to ensure good timing
    trig = stim_client.get_trigger(timeout=0.2)

    # wait till 0.75 sec elapses
    while timer1.getTime() < 0:
        pass

    fixation.draw()  # draw fixation
    mywin.flip()  # show fixation dot

    timer2.reset()  # reset timer
    timer2.add(0.25)  # display stimuli for 0.25 sec

    # display fixation cross for 0.25 seconds
    while timer2.getTime() < 0:
        pass

mywin.close()  # close the window
core.quit()
