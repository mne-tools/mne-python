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

# The event-id list for first 10 stimuli
# The rest will be decided on the fly
ev_list = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]

for ii in range(50):

    if ii > 10:
        trig = stim_client.get_trigger()
    else:
        trig = ev_list[ii]

    if trig == 1:
        right_cb.draw()
    else:
        left_cb.draw()

    mywin.update()

    #pause, so you get a chance to see it!
    core.wait(1.0)

core.quit()
mywin.close()
