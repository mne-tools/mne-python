"""
.. _ex-plot-events:

======================================
Using the event system to link figures
======================================

Many of MNE-Python's figures are interactive. For example, you can select channels or
scroll through time. The event system allows you to link figures together so that
interacting with one figure will simultaneously update another figure.

In this example, we'll be linking two topomap plots, such that selecting the
time in one will also update the time in the other.

Since the figures on our website don't have any interaction capabilities, this example
will only work properly when run in an interactive environment.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause

########################################################################################
# Load some evoked and source data to plot.
import mne

data_path = mne.datasets.sample.data_path()
evokeds = mne.read_evokeds(data_path / "MEG" / "sample" / "sample_audvis-ave.fif")
for ev in evokeds:
    ev.apply_baseline()

########################################################################################
# Plot topomaps of two experimental conditions. Then link the figures together,
# so they can communicate. What kind of information is communicated between
# figures depends on the figure types. In this case, the information about the
# currently selected time is shared.
fig1 = evokeds[0].plot_topomap("interactive")
fig2 = evokeds[1].plot_topomap("interactive")
mne.viz.ui_events.link(fig1, fig2)
