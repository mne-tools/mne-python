"""
.. _tut_viz_evoked:

=====================
Visualize Evoked data
=====================
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
evoked_laud = evoked[0]
evoked_raud = evoked[1]
evoked_lvis = evoked[2]
evoked_rvis = evoked[3]

###############################################################################
# Let's start with a simple one. Here we plot event related potentials / fields
# (ERP/ERF). All plotting functions of MNE-python return a handle to the figure
# instance. When we have the handle, we can customise the plots to our liking.
fig = evoked_laud.plot()

# We can get rid of the empty space with a simple function call.
fig.tight_layout()

###############################################################################
# Now let's make it a bit fancier and only use MEG channels. Many of the
# MNE-functions include a ``picks`` parameter to include a selection of
# channels. ``picks`` is simply a list of channel indices that you can easily
# construct with :func:`mne.pick_types`. See also :func:`mne.pick_channels` and
# :func:`mne.pick_channels_regexp`.
picks = mne.pick_types(evoked_laud.info, meg=True, eeg=False, eog=False)
evoked_laud.plot(spatial_colors=True, gfp=True, picks=picks)

###############################################################################
# Notice the legend on the left. The colors would suggest that there may be two
# separate sources for the signals. This wasn't obvious from the first figure.
# Try painting the slopes with left mouse button. It should open a new window
# with topomaps (scalp plots) of the average over the painted area. There is
# also a function for drawing topomaps separately.
evoked_raud.plot_topomap()

###############################################################################
# By default the topomaps are drawn from evenly spread out points of time over
# the evoked data. We can also define the times ourselves.
times = np.arange(0.05, 0.151, 0.05)
evoked_raud.plot_topomap(times=times, ch_type='mag')

###############################################################################
# Or we can automatically select the peaks.
evoked_raud.plot_topomap(times='peaks', ch_type='mag')

###############################################################################
# You can take a look at the documentation of :func:`mne.Evoked.plot_topomap`
# or simply write ``evoked_raud.plot_topomap?`` in your python console to
# see the different parameters you can pass to this function. Most of the
# plotting functions also accept ``axes`` parameter. With that, you can
# customise your plots even further. First we shall create a set of matplotlib
# axes in a single figure and plot all of our evoked categories next to each
# other.
fig, ax = plt.subplots(1, 5)
evoked_laud.plot_topomap(times=0.1, axes=ax[0], show=False)
evoked_raud.plot_topomap(times=0.1, axes=ax[1], show=False)
evoked_lvis.plot_topomap(times=0.1, axes=ax[2], show=False)
evoked_rvis.plot_topomap(times=0.1, axes=ax[3], show=True)

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
# key-value pairs as a python dictionary that gets passed as parameters to the
# topomaps of the joint plot. Here we disable the plotting of sensor locations
# in the topomaps
topomap_args = dict(sensors=False)
evoked_raud.plot_joint(topomap_args=topomap_args)

###############################################################################
# We can also plot the activations as images. The time runs along the x-axis
# and the channels along the y-axis. The amplitudes are color coded so that
# the amplitudes from negative to positive translates to shift from blue to
# red. White means zero amplitude. You can use the ``cmap`` parameter to define
# the color map yourself. The accepted values include all matplotlib colormaps.
evoked_laud.plot_image(picks=picks)

###############################################################################
# Finally we plot the sensor data as a topographical view. In the simple case
# we plot only left auditory responses, and then we plot them all in the same
# figure for comparison. Click on the individual plots to open them bigger.
title = 'MNE sample data (condition : %s)'
evoked_laud.plot_topo(title=title % evoked_laud.comment)
colors = 'yellow', 'green', 'red', 'blue'
mne.viz.plot_evoked_topo(evoked, color=colors,
                         title=title % 'Left/Right Auditory/Visual')

###############################################################################
# Visualizing field lines in 3D
# -----------------------------
#
# We now compute the field maps to project MEG and EEG data to MEG helmet
# and scalp surface.
#
# To do this we'll need coregistration information. See
# :ref:`tut_forward` for more details.
#
# Here we just illustrate usage.

subjects_dir = data_path + '/subjects'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

maps = mne.make_field_map(evoked_laud, trans=trans_fname, subject='sample',
                          subjects_dir=subjects_dir, n_jobs=1)

# explore several points in time
field_map = evoked_laud.plot_field(maps, time=.1)

###############################################################################
# .. note::
#       If trans_fname is set to None then only MEG estimates can be visualized
