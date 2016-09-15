"""
.. _tut_viz_epochs:

Visualize Epochs data
=====================

"""
import os.path as op

import mne

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
                          add_eeg_ref=False)
raw.set_eeg_reference()  # set EEG average reference
events = mne.read_events(op.join(data_path, 'sample_audvis_raw-eve.fif'))
picks = mne.pick_types(raw.info, meg='grad')
epochs = mne.Epochs(raw, events, [1, 2], picks=picks, add_eeg_ref=False)

###############################################################################
# This tutorial focuses on visualization of epoched data. All of the functions
# introduced here are basically high level matplotlib functions with built in
# intelligence to work with epoched data. All the methods return a handle to
# matplotlib figure instance.
#
# All plotting functions start with ``plot``. Let's start with the most
# obvious. :func:`mne.Epochs.plot` offers an interactive browser that allows
# rejection by hand when called in combination with a keyword ``block=True``.
# This blocks the execution of the script until the browser window is closed.
epochs.plot(block=True)

###############################################################################
# The numbers at the top refer to the event id of the epoch. We only have
# events with id numbers of 1 and 2 since we included only those when
# constructing the epochs.
#
# Since we did no artifact correction or rejection, there are epochs
# contaminated with blinks and saccades. For instance, epoch number 9 (see
# numbering at the bottom) seems to be contaminated by a blink (scroll to the
# bottom to view the EOG channel). This epoch can be marked for rejection by
# clicking on top of the browser window. The epoch should turn red when you
# click it. This means that it will be dropped as the browser window is closed.
# You should check out `help` at the lower left corner of the window for more
# information about the interactive features.
#
# To plot individual channels as an image, where you see all the epochs at one
# glance, you can use function :func:`mne.Epochs.plot_image`. It shows the
# amplitude of the signal over all the epochs plus an average of the
# activation. We explicitly set interactive colorbar on (it is also on by
# default for plotting functions with a colorbar except the topo plots). In
# interactive mode you can scale and change the colormap with mouse scroll and
# up/down arrow keys. You can also drag the colorbar with left/right mouse
# button. Hitting space bar resets the scale.
epochs.plot_image(97, cmap='interactive')

# You also have functions for plotting channelwise information arranged into a
# shape of the channel array. The image plotting uses automatic scaling by
# default, but noisy channels and different channel types can cause the scaling
# to be a bit off. Here we define the limits by hand.
epochs.plot_topo_image(vmin=-200, vmax=200, title='ERF images')
