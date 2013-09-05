"""
==============================================
Real-time feedback for decoding :: Client Side
==============================================

This example demonstrates how to setup a real-time feedback
mechanism using StimServer and StimClient.

"""

print __doc__

# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from mne.realtime import StimClient
from psychopy import visual, core

# Instantiating stimulation client
stim_client = StimClient('localhost', port=4218)

#create a window
mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")

#create the stimuli
right_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                             visibleWedge=[0, 180], radialCycles=4,
                             angularCycles=8, interpolate=False,
                             autoLog=False)

left_cb = visual.RadialStim(mywin, tex='sqrXsqr', color=1, size=5,
                            visibleWedge=[180, 360], radialCycles=4,
                            angularCycles=8, interpolate=False,
                            autoLog=False)

fixation = visual.PatchStim(mywin, color=-1, colorSpace='rgb', tex=None,
                            mask='circle', size=0.2)

# the most accurate method is using frame refresh periods
# however, since the actual refresh rate is not known
# we use the Clock
timer1 = core.Clock()
timer2 = core.Clock()

ev_list = list()
trig = 4

# iterating over 50 epochs
for ii in range(50):

    if trig is not None:
        ev_list.append(trig)
    else:
        ev_list.append(ev_list[-1])  # use the last stimuli

    if ev_list[ii] == 3:
        left_cb.draw()
    else:
        right_cb.draw()

    mywin.flip()

    timer1.reset()
    timer1.add(0.75)  # display stimuli for 0.75 sec

    # return within 0.2 seconds (< 0.75 seconds) to ensure good timing
    trig = stim_client.get_trigger(timeout=0.2)

    # wait till 0.75 sec elapses
    while timer1.getTime() < 0:
        pass

    fixation.draw()
    mywin.flip()

    timer2.reset()
    timer2.add(0.25)

    # display fixation cross for 0.25 seconds
    while timer2.getTime() < 0:
        pass

mywin.close()
core.quit()
