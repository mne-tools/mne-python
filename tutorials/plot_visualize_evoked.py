"""
.. _visualize_evoked:

==========================
Visualizing evoked dataset
==========================
"""
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne

###############################################################################
# In this tutorial we focus on plotting functions of :class:`mne.Evoked`.
# Here we read the evoked object from a file. Check out
# :ref:`tut_epoching_and_averaging` to get to this stage from raw data.
data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
evoked = mne.read_evokeds(fname, baseline=(None, 0), proj=True)
print(evoked)

###############################################################################
# Notice that ``evoked`` is a list of evoked instances. You can read only one
# of the categories by passing the argument ``condition`` to
# :func:`mne.read_evokeds`. To make things more simple for this tutorial, we
# read each instance to a variable.
left_auditory = evoked[0]
right_auditory = evoked[1]
left_visual = evoked[2]
right_visual = evoked[3]

###############################################################################
# Let's start with a simple one.
fig = left_auditory.plot()

###############################################################################
# All plotting functions of MNE-python returns a handle to the figure instance.
# When we have the handle, we can customise the plots to our liking. We can get
# rid of the empty space with a simple function call.
fig.tight_layout()

###############################################################################
# Now let's make it a bit fancier and only use MEG channels. Many of the
# MNE-functions include a ``picks`` parameter to include a selection of
# channels. ``picks`` is simply a list of channel indices that you can easily
# construct with :func:`mne.pick_types`. See also :func:`mne.pick_channels` and
# :func:`mne.pick_channels_regexp`.
picks = mne.pick_types(left_auditory.info, meg=True, eeg=False, eog=False)
left_auditory.plot(spatial_colors=True, gfp=True, picks=picks)

###############################################################################
# Notice the legend on the left. The colors would suggest that there may be two
# separate sources for the signals. This wasn't obvious from the first figure.
# Try painting the slopes with left mouse button. It should open a new window
# with topomaps (scalp plots) of the painted area. There is also a function
# for drawing topomaps separately.
right_auditory.plot_topomap()

###############################################################################
# By default the topomaps are drawn from evenly spread out points of time over
# the evoked data. We can also define the times ourselves.
times = np.arange(0.05, 0.151, 0.05)
right_auditory.plot_topomap(times=times, ch_type='mag')

###############################################################################
# Or we can automatically select the peaks.
right_auditory.plot_topomap(times='peaks', ch_type='mag')

###############################################################################
# You can take a look at the documentation of :func:`mne.Evoked.plot_topomap`
# or simply write ``right_auditory.plot_topomap?`` in your python console to
# see the different parameters you can pass to this function. Most of the
# plotting functions also accept ``axes`` parameter. With that, you can
# customise your plots even further. First we shall create a set of matplotlib
# axes in a single figure and plot all of our evoked categories next to each
# other.
fig, ax = plt.subplots(1, 5)
left_auditory.plot_topomap(times=0.1, axes=ax[0], show=False)
right_auditory.plot_topomap(times=0.1, axes=ax[1], show=False)
left_visual.plot_topomap(times=0.1, axes=ax[2], show=False)
right_visual.plot_topomap(times=0.1, axes=ax[3], show=True)

###############################################################################
# Notice that we created five axes, but had only four categories. The fifth
# axes was used for drawing the colorbar. You must provide room for it when you
# create this kind of custom plots or turn the colorbar off with
# ``colorbar=False``. That's what the warnings are trying to tell you. Also, we
# used ``show=False`` for the three first function calls. This prevents the
# showing of the figure prematurely. The behavior depends on the mode you are
# using for your python session. See http://matplotlib.org/users/shell.html for
# more information.
#
# We can also combine the two kinds of plots to one. Notice the
# ``topomap_args`` parameter of :func:`mne.Evoked.plot_joint`. You can pass
# key-value pairs as a python dictionary that gets passed to the topomaps of
# the joint plot.
right_auditory.plot_joint()

###############################################################################
# Finally we plot the sensor data as a topographical view. In the simple case
# we plot only the ``left_auditory``, and then we plot them all in the same
# figure for comparison. Click on the individual plots to open them bigger.
left_auditory.plot_topo()
colors = 'yellow', 'green', 'red', 'blue'
mne.viz.plot_evoked_topo(evoked, color=colors)
