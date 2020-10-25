"""
.. _ex-customize-mne-viz:

============================
Customize MNE visualizations
============================

In this example, we show how to take MNE plots and customize them
using Matplotlib.

To get an idea on how to makea figure closer to publication-ready,
take a look at :ref:`ex-publication-figure`.
"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

import mne

###############################################################################
# Suppose we want a figure with some mean timecourse extracted from a number of
# sensors, and we want a smaller panel within the figure to show a head outline
# with the positions of those sensors clearly marked.
# If you are familiar with MNE, you know that this is something that
# :func:`mne.viz.plot_compare_evokeds` does, see an example output in
# :ref:`ex-hf-sef-data` at the bottom.
#
# In this example we will show you how to achieve this result on your own
# figure, without having to use :func:`mne.viz.plot_compare_evokeds`!
#
# Let's start by loading some :ref:`example data <sample-dataset>`.

data_path = mne.datasets.sample.data_path()
fname_raw = op.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
raw = mne.io.read_raw_fif(fname_raw)

# For the sake of the example, we focus on EEG data
raw.pick_types(meg=False, eeg=True)


###############################################################################
# Let's make a plot.

# channels to plot:
to_plot = [f"EEG {i:03}" for i in range(1, 5)]

# get the data for plotting in a short time interval from 10 to 20 seconds
start = int(raw.info['sfreq'] * 10)
stop = int(raw.info['sfreq'] * 20)
data, times = raw.get_data(picks=to_plot,
                           start=start, stop=stop, return_times=True)

# Scale the data from the MNE internal unit V to µV
data *= 1e6

# Take the mean of the channels
mean = np.mean(data, axis=0)

# make a figure
fig, ax = plt.subplots()

# plot some fake EEG data
ax.plot(times, mean)
ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)")

###############################################################################
# So far so good. Now let's add the smaller figure within the figure to show
# exactly, which sensors we used to make the timecourse.

# For that, we use an "inset_axes" that we plot into our existing axes
axins = inset_axes(ax, width="30%", height="30%", loc=2)

# The head outline with the sensor positions can be plotted using the
# MNE :class:`mne.io.raw` object that is the origin of our data.
# Specifically, that object already contains all the sensor positions,
# and we can plot them using the ``plot_sensors`` method.
raw.copy().pick_channels(to_plot).plot_sensors(
    title="", axes=axins
)

###############################################################################
# That looks nice. But the sensor dots are way too big for our taste.
# Luckily, all MNE plots use Matplotlib under the hood and we can customize
# each and every facet of them.
# To make the sensor dots smaller, we need to first get a handle on them to
# then apply a ``*.set_*`` method on them.

# If we inspect our axes we find the objects contained in our plot:
print(axins.get_children())

# That's quite a mess, but we know that we want to change the sensor dots,
# and those are most certainly a "PathCollection" object.
print(axins.collections)

# Now we we found exactly what we needed. Sometimes this can take a bit of
# experimentation.
sensor_dots = axins.collections[0]

# Let's shrink the sensor dots to finish our figure.
sensor_dots.set_sizes([1])

