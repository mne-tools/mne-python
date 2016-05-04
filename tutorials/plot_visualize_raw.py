"""
.. _tut_viz_raw:

Visualize Raw data
==================

"""
import os.path as op

import mne

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'))
events = mne.read_events(op.join(data_path, 'sample_audvis_raw-eve.fif'))

###############################################################################
# The visualization module (:mod:`mne.viz`) contains all the plotting functions
# that work in combination with MNE data structures. Usually the easiest way to
# use them is to call a method of the data container. All of the plotting
# method names start with ``plot``. If you're using Ipython console, you can
# just write ``raw.plot`` and ask the interpreter for suggestions with a
# ``tab`` key.
#
# To visually inspect your raw data, you can use the python equivalent of
# ``mne_browse_raw``.
raw.plot(block=True, events=events)

###############################################################################
# The channels are color coded by channel type. Generally MEG channels are
# colored in different shades of blue, whereas EEG channels are black. The
# channels are also sorted by channel type by default. If you want to use a
# custom order for the channels, you can use ``order`` parameter of
# :func:`raw.plot`. The scrollbar on right side of the browser window also
# tells us that two of the channels are marked as ``bad``. Bad channels are
# color coded gray. By clicking the lines or channel names on the left, you can
# mark or unmark a bad channel interactively. You can use +/- keys to adjust
# the scale (also = works for magnifying the data). Note that the initial
# scaling factors can be set with parameter ``scalings``. If you don't know the
# scaling factor for channels, you can automatically set them by passing
# scalings='auto'. With ``pageup/pagedown`` and ``home/end`` keys you can
# adjust the amount of data viewed at once. To see all the interactive
# features, hit ``?`` or click ``help`` in the lower left corner of the
# browser window.
#
# We read the events from a file and passed it as a parameter when calling the
# method. The events are plotted as vertical lines so you can see how they
# align with the raw data.
#
# We can check where the channels reside with ``plot_sensors``. Notice that
# this method (along with many other MNE plotting functions) is callable using
# any MNE data container where the channel information is available.
raw.plot_sensors(kind='3d', ch_type='mag')

###############################################################################
# Now let's add some ssp projectors to the raw data. Here we read them from a
# file and plot them.
projs = mne.read_proj(op.join(data_path, 'sample_audvis_eog-proj.fif'))
raw.add_proj(projs)
raw.plot_projs_topomap()

###############################################################################
# The first three projectors that we see are the SSP vectors from empty room
# measurements to compensate for the noise. The fourth one is the average EEG
# reference. These are already applied to the data and can no longer be
# removed. The next six are the EOG projections that we added. Every data
# channel type has two projection vectors each. Let's try the raw browser
# again.
raw.plot()

###############################################################################
# Now click the `proj` button at the lower right corner of the browser
# window. A selection dialog should appear, where you can toggle the projectors
# on and off. Notice that the first four are already applied to the data and
# toggling them does not change the data. However the newly added projectors
# modify the data to get rid of the EOG artifacts. Note that toggling the
# projectors here doesn't actually modify the data. This is purely for visually
# inspecting the effect. See :func:`mne.io.Raw.del_proj` to actually remove the
# projectors.
#
# Raw container also lets us easily plot the power spectra over the raw data.
# See the API documentation for more info.
raw.plot_psd()

###############################################################################
# Plotting channel wise power spectra is just as easy. The layout is inferred
# from the data by default when plotting topo plots. This works for most data,
# but it is also possible to define the layouts by hand. Here we select a
# layout with only magnetometer channels and plot it. Then we plot the channel
# wise spectra of first 30 seconds of the data.
layout = mne.channels.read_layout('Vectorview-mag')
layout.plot()
raw.plot_psd_topo(tmax=30., fmin=5., fmax=60., n_fft=1024, layout=layout)
