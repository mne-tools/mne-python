"""
.. _ex-plot-events:

======================================
Using the event system to link figures
======================================

Many of MNE-Python's figures are interactive. For example, you can select channels or
scroll through time. The event system allows you to link figures together so that
interacting with one figure will simultaneously update another figure.

In this example, we'll be linking an evoked plot to a source estimate plot, such that
selecting the time in one will also update the time in the other.

Since the figures on our website don't have any interaction capabilities, this example
will only work properly when run in an interactive environment.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD-3-Clause

# %%
# Load some evoked and source data to plot.
import matplotlib.pyplot as plt
import mne

data_path = mne.datasets.sample.data_path()
evoked = mne.read_evokeds(
    data_path / "MEG" / "sample" / "sample_audvis-ave.fif", condition="Left Auditory"
)
evoked.apply_baseline()
stc = mne.read_source_estimate(data_path / "MEG" / "sample" / "sample_audvis-meg-eeg")
evoked.crop(0, stc.times[-1])

# %%
# Enable interactivity. I'm not sure exactly why we need this.
plt.ion()

# %%
# Plot both the source estimate plot, with time interaction enabled, and a sensor-level
# topomap. Then link the figures together, so they can communicate. What kind of
# information is communicated between figures depends on the figure types. In this case,
# the information about the currently selected time is shared.
fig1 = stc.plot("sample", subjects_dir=data_path / "subjects", initial_time=0.1)
fig1.set_time_interpolation('linear')
fig2 = evoked.plot_topomap('interactive')
mne.viz.ui_events.link(fig1, fig2)
