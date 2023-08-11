"""
.. _ex-plot-events:

======================================
Using the event system to link figures
======================================

Many of MNE-Python's figures are interactive. For example, you can select channels or
scroll through time. The event system allows you to link figures together so that
interacting with one figure will simultaneously update another figure.

In this example, we'll be looking at linking two topomap plots, such that selecting the
time in one will also update the time in the other, as well as hooking our own
custom plot into MNE-Python's event system.

Since the figures on our website don't have any interaction capabilities, this example
will only work properly when run in an interactive environment.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause
import mne
import matplotlib.pyplot as plt
from mne.viz.ui_events import publish, subscribe, link, TimeChange

# Turn on interactivity
plt.ion()

########################################################################################
# Linking interactive plots
# =========================
# We load evoked data for two experimental conditions and create two topomap
# plots that have sliders controlling the time-point that is shown. By default,
# both figures are independent, but we will link the event channels of the
# figures together, so that moving the slider in one figure will also move the
# slider in the other.
data_path = mne.datasets.sample.data_path()
fname = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"
aud_left = mne.read_evokeds(fname, condition="Left Auditory").apply_baseline()
aud_right = mne.read_evokeds(fname, condition="Right Auditory").apply_baseline()
fig1 = aud_left.plot_topomap("interactive")
fig2 = aud_right.plot_topomap("interactive")
link(fig1, fig2)  # link the event channels

########################################################################################
# Hooking a custom plot into the event system
# ===========================================
# In MNE-Python, each figure has an associated event channel. Drawing routines
# can :func:`publish <mne.viz.ui_events.publish>` events on the channel and
# receive events by :func:`subscribe <mne.viz.ui_events.subscribe>`-ing to the
# channel. When subscribing to an event on a channel, you specify a callback
# function is will be called whenever a drawing routine publishes that event on
# the event channel.
#
# The events are modeled after matplotlib's event system. Each event has a string
# name (the snake-case version of its class name) and a list of relevant values.
# For example, the "time_change" event should have the new time as a value.
# Values can be any python object. When publishing an event, the publisher
# creates a new instance of the event's class. When subscribing to an event,
# having to dig up and import the correct class is a bit of a hassle. Following
# matplotlib's example, subscribers use the string name of the event to refer
# to it.
#
# Below, we create a custom plot and then make it publish and subscribe to
# :class:`TimeChange <mne.viz.ui_events.TimeChange>` events so it can work
# together with the topomap plots we created earlier.
fig3, ax = plt.subplots()
ax.plot(aud_left.times, aud_left.pick("mag").data.max(axis=0), label="Left")
ax.plot(aud_right.times, aud_right.pick("mag").data.max(axis=0), label="Right")
time_bar = ax.axvline(0, color="black")  # Our time slider
ax.set_xlabel("Time (s)")
ax.set_ylabel("Maximum magnetic field strength")
ax.set_title("A custom plot")
plt.legend()


def on_motion_notify(mpl_event):
    """Respond to matplotlib's mouse event.

    Publishes an MNE-Python TimeChange event. When the mouse goes out of
    bounds, the xdata will be None, which is a special case that needs to be
    handled.
    """
    if mpl_event.xdata is not None:
        publish(fig3, TimeChange(time=mpl_event.xdata))


def on_time_change(event):
    """Respond to MNE-Python's TimeChange event. Updates the plot."""
    time_bar.set_xdata([event.time])
    fig3.canvas.draw()  # update the figure


plt.connect("motion_notify_event", on_motion_notify)
subscribe(fig3, "time_change", on_time_change)

# Link the new figure with the topomap plots, so that the TimeChange events are
# send to all of them.
link(fig3, fig1)
link(fig3, fig2)
