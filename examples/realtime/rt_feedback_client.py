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

timer1 = core.Clock()
timer2 = core.Clock()

for ii in range(50):

    timer2.reset()
    timer2.add(1.0)  # time between stimuli

    trig = stim_client.get_trigger()

    if trig == 4:
        right_cb.draw()
    else:
        left_cb.draw()

    mywin.update()

    timer1.reset()
    timer1.add(0.75)  # display stimuli for 0.75 sec

    while timer1.getTime() < 0:
        pass

    fixation.draw()
    mywin.update()

    while timer2.getTime() < 0:
        pass

core.quit()
mywin.close()
