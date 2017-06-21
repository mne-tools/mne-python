"""
.. _tut_viz_epochs:

Visualize Epochs data
=====================

"""
import os.path as op

import mne

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'))
raw.set_eeg_reference()  # set EEG average reference
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
events = mne.read_events(op.join(data_path, 'sample_audvis_raw-eve.fif'))
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=1.)

###############################################################################
# This tutorial focuses on visualization of epoched data. All of the functions
# introduced here are basically high level matplotlib functions with built in
# intelligence to work with epoched data. All the methods return a handle to
# matplotlib figure instance.
#
# Events used for constructing the epochs here are the triggers for subject
# being presented a smiley face at the center of the visual field. More of the
# paradigm at :ref:`BABDHIFJ`.
#
# All plotting functions start with ``plot``. Let's start with the most
# obvious. :func:`mne.Epochs.plot` offers an interactive browser that allows
# rejection by hand when called in combination with a keyword ``block=True``.
# This blocks the execution of the script until the browser window is closed.
epochs.plot(block=True)

###############################################################################
# The numbers at the top refer to the event id of the epoch. The number at the
# bottom is the running numbering for the epochs.
#
# Since we did no artifact correction or rejection, there are epochs
# contaminated with blinks and saccades. For instance, epoch number 1 seems to
# be contaminated by a blink (scroll to the bottom to view the EOG channel).
# This epoch can be marked for rejection by clicking on top of the browser
# window. The epoch should turn red when you click it. This means that it will
# be dropped as the browser window is closed.
#
# It is possible to plot event markers on epoched data by passing ``events``
# keyword to the epochs plotter. The events are plotted as vertical lines and
# they follow the same coloring scheme as :func:`mne.viz.plot_events`. The
# events plotter gives you all the events with a rough idea of the timing.
# Since the colors are the same, the event plotter can also function as a
# legend for the epochs plotter events. It is also possible to pass your own
# colors via ``event_colors`` keyword. Here we can plot the reaction times
# between seeing the smiley face and the button press (event 32).
#
# When events are passed, the epoch numbering at the bottom is switched off by
# default to avoid overlaps. You can turn it back on via settings dialog by
# pressing `o` key. You should check out `help` at the lower left corner of the
# window for more information about the interactive features.
events = mne.pick_events(events, include=[5, 32])
mne.viz.plot_events(events)
epochs['smiley'].plot(events=events)

###############################################################################
# To plot individual channels as an image, where you see all the epochs at one
# glance, you can use function :func:`mne.Epochs.plot_image`. It shows the
# amplitude of the signal over all the epochs plus an average (evoked response)
# of the activation. We explicitly set interactive colorbar on (it is also on
# by default for plotting functions with a colorbar except the topo plots). In
# interactive mode you can scale and change the colormap with mouse scroll and
# up/down arrow keys. You can also drag the colorbar with left/right mouse
# button. Hitting space bar resets the scale.
epochs.plot_image(278, cmap='interactive')

###############################################################################
# You also have functions for plotting channelwise information arranged into a
# shape of the channel array. The image plotting uses automatic scaling by
# default, but noisy channels and different channel types can cause the scaling
# to be a bit off. Here we define the limits by hand.
epochs.plot_topo_image(vmin=-200, vmax=200, title='ERF images')
