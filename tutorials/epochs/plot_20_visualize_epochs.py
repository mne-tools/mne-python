"""
.. _tut-visualize-epochs:

Visualizing epoched data
========================

This tutorial shows how to plot epoched data as time series, how to plot the
spectral density of epoched data, how to plot epochs as an imagemap, and how to
plot the sensor locations and projectors stored in :class:`~mne.Epochs`
objects.

.. contents:: Page contents
   :local:
   :depth: 2

We'll start by importing the modules we need, loading the continuous (raw)
sample data, and cropping it to save memory:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=120)

###############################################################################
# To create the :class:`~mne.Epochs` data structure, we'll extract the event
# IDs stored in the :term:`stim channel`, map those integer event IDs to more
# descriptive condition labels using an event dictionary, and pass those to the
# :class:`~mne.Epochs` constructor, along with the :class:`~mne.io.Raw` data
# and the desired temporal limits of our epochs, ``tmin`` and ``tmax`` (for a
# detailed explanation of these steps, see :ref:`tut-epochs-class`).

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, event_id=event_dict,
                    preload=True)
del raw

###############################################################################
# Plotting ``Epochs`` as time series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: Interactivity in pipelines and scripts
#
#     To use the interactive features of the :meth:`~mne.Epochs.plot` method
#     when running your code non-interactively, pass the ``block=True``
#     parameter, which halts the Python interpreter until the figure window is
#     closed. That way, any channels or epochs that you mark as "bad" will be
#     taken into account in subsequent processing steps.
#
# To visualize epoched data as time series (one time series per channel), the
# :meth:`mne.Epochs.plot` method is available. It creates an interactive window
# where you can scroll through epochs and channels, enable/disable any
# unapplied :term:`SSP projectors <projector>` to see how they affect the
# signal, and even manually mark bad channels (by clicking the channel name) or
# bad epochs (by clicking the data) for later dropping. Channels marked "bad"
# will be shown in light grey color and will be added to
# ``epochs.info['bads']``; epochs marked as bad will be indicated as ``'USER'``
# in ``epochs.drop_log``.
#
# Here we'll plot only the "catch" trials from the :ref:`sample dataset
# <sample-dataset>`, and pass in our events array so that the button press
# responses also get marked (we'll plot them in red, and plot the "face" events
# defining time zero for each epoch in blue). We also need to pass in
# our ``event_dict`` so that the :meth:`~mne.Epochs.plot` method will know what
# we mean by "buttonpress" — this is because subsetting the conditions by
# calling ``epochs['face']`` automatically purges the dropped entries from
# ``epochs.event_id``:

catch_trials_and_buttonpresses = mne.pick_events(events, include=[5, 32])
epochs['face'].plot(events=catch_trials_and_buttonpresses, event_id=event_dict,
                    event_colors=dict(buttonpress='red', face='blue'))

###############################################################################
# Plotting projectors from an ``Epochs`` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the plot above we can see heartbeat artifacts in the magnetometer
# channels, so before we continue let's load ECG projectors from disk and apply
# them to the data:

ecg_proj_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                             'sample_audvis_ecg-proj.fif')
ecg_projs = mne.read_proj(ecg_proj_file)
epochs.add_proj(ecg_projs)
epochs.apply_proj()

###############################################################################
# Just as we saw in the :ref:`tut-section-raw-plot-proj` section, we can plot
# the projectors present in an :class:`~mne.Epochs` object using the same
# :meth:`~mne.Epochs.plot_projs_topomap` method. Since the original three
# empty-room magnetometer projectors were inherited from the
# :class:`~mne.io.Raw` file, and we added two ECG projectors for each sensor
# type, we should see nine projector topomaps:

epochs.plot_projs_topomap(vlim='joint')

###############################################################################
# Note that these field maps illustrate aspects of the signal that *have
# already been removed* (because projectors in :class:`~mne.io.Raw` data are
# applied by default when epoching, and because we called
# :meth:`~mne.Epochs.apply_proj` after adding additional ECG projectors from
# file). You can check this by examining the ``'active'`` field of the
# projectors:

print(all(proj['active'] for proj in epochs.info['projs']))

###############################################################################
# Plotting sensor locations
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Just like :class:`~mne.io.Raw` objects, :class:`~mne.Epochs` objects keep
# track of sensor locations, which can be visualized with the
# :meth:`~mne.Epochs.plot_sensors` method:

epochs.plot_sensors(kind='3d', ch_type='all')
epochs.plot_sensors(kind='topomap', ch_type='all')

###############################################################################
# Plotting the power spectrum of ``Epochs``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Again, just like :class:`~mne.io.Raw` objects, :class:`~mne.Epochs` objects
# have a :meth:`~mne.Epochs.plot_psd` method for plotting the `spectral
# density`_ of the data.

epochs['auditory'].plot_psd(picks='eeg')

###############################################################################
# Plotting ``Epochs`` as an image map
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A convenient way to visualize many epochs simultaneously is to plot them as
# an image map, with each row of pixels in the image representing a single
# epoch, the horizontal axis representing time, and each pixel's color
# representing the signal value at that time sample for that epoch. Of course,
# this requires either a separate image map for each channel, or some way of
# combining information across channels. The latter is possible using the
# :meth:`~mne.Epochs.plot_image` method; the former can be achieved with the
# :meth:`~mne.Epochs.plot_image` method (one channel at a time) or with the
# :meth:`~mne.Epochs.plot_topo_image` method (all sensors at once).
#
# By default, the image map generated by :meth:`~mne.Epochs.plot_image` will be
# accompanied by a scalebar indicating the range of the colormap, and a time
# series showing the average signal across epochs and a bootstrapped 95%
# confidence band around the mean. :meth:`~mne.Epochs.plot_image` is a highly
# customizable method with many parameters, including customization of the
# auxiliary colorbar and averaged time series subplots. See the docstrings of
# :meth:`~mne.Epochs.plot_image` and `mne.viz.plot_compare_evokeds` (which is
# used to plot the average time series) for full details. Here we'll show the
# mean across magnetometers for all epochs with an auditory stimulus:

epochs['auditory'].plot_image(picks='mag', combine='mean')

###############################################################################
# To plot image maps for individual sensors or a small group of sensors, use
# the ``picks`` parameter. Passing ``combine=None`` (the default) will yield
# separate plots for each sensor in ``picks``; passing ``combine='gfp'`` will
# plot the global field power (useful for combining sensors that respond with
# opposite polarity).

# sphinx_gallery_thumbnail_number = 8
epochs['auditory'].plot_image(picks=['MEG 0242', 'MEG 0243'])
epochs['auditory'].plot_image(picks=['MEG 0242', 'MEG 0243'], combine='gfp')

###############################################################################
# To plot an image map for *all* sensors, use
# :meth:`~mne.Epochs.plot_topo_image`, which is optimized for plotting a large
# number of image maps simultaneously, and (in interactive sessions) allows you
# to click on each small image map to pop open a separate figure with the
# full-sized image plot (as if you had called :meth:`~mne.Epochs.plot_image` on
# just that sensor). At the small scale shown in this tutorial it's hard to see
# much useful detail in these plots; it's often best when plotting
# interactively to maximize the topo image plots to fullscreen. The default is
# a figure with black background, so here we specify a white background and
# black foreground text. By default :meth:`~mne.Epochs.plot_topo_image` will
# show magnetometers and gradiometers on the same plot (and hence not show a
# colorbar, since the sensors are on different scales) so we'll also pass a
# :class:`~mne.channels.Layout` restricting each plot to one channel type.
# First, however, we'll also drop any epochs that have unusually high signal
# levels, because they can cause the colormap limits to be too extreme and
# therefore mask smaller signal fluctuations of interest.

reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=150e-6)       # 150 μV
epochs.drop_bad(reject=reject_criteria)

for ch_type, title in dict(mag='Magnetometers', grad='Gradiometers').items():
    layout = mne.channels.find_layout(epochs.info, ch_type=ch_type)
    epochs['auditory/left'].plot_topo_image(layout=layout, fig_facecolor='w',
                                            font_color='k', title=title)

###############################################################################
# To plot image maps for all EEG sensors, pass an EEG layout as the ``layout``
# parameter of :meth:`~mne.Epochs.plot_topo_image`. Note also here the use of
# the ``sigma`` parameter, which smooths each image map along the vertical
# dimension (across epochs) which can make it easier to see patterns across the
# small image maps (by smearing noisy epochs onto their neighbors, while
# reinforcing parts of the image where adjacent epochs are similar). However,
# ``sigma`` can also disguise epochs that have persistent extreme values and
# maybe should have been excluded, so it should be used with caution.

layout = mne.channels.find_layout(epochs.info, ch_type='eeg')
epochs['auditory/left'].plot_topo_image(layout=layout, fig_facecolor='w',
                                        font_color='k', sigma=1)

###############################################################################
# .. LINKS
#
# .. _spectral density: https://en.wikipedia.org/wiki/Spectral_density
