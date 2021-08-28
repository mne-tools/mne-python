"""
.. _tut-erp:

EEG processing and Event Related Potentials (ERPs)
==================================================

This tutorial shows how to perform standard ERP analyses in MNE-Python. Most of
the material here is covered in other tutorials too, but for convenience the
functions and methods most useful for ERP analyses are collected here, with
links to other tutorials where more detailed information is given.

As usual we'll start by importing the modules we need and loading some example
data. Instead of parsing the events from the raw data's :term:`stim channel`
(like we do in :ref:`this tutorial <tut-events-vs-annotations>`), we'll load
the events from an external events file. Finally, to speed up computations so
our documentation server can handle them, we'll crop the raw data from ~4.5
minutes down to 90 seconds.
"""

# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_filt-0-40_raw-eve.fif')
events = mne.read_events(sample_data_events_file)

raw.crop(tmax=90)  # in seconds; happens in-place
# discard events >90 seconds (not strictly necessary: avoids some warnings)
events = events[events[:, 0] <= raw.last_samp]

# %%
# The file that we loaded has already been partially processed: 3D sensor
# locations have been saved as part of the ``.fif`` file, the data have been
# low-pass filtered at 40 Hz, and a common average reference is set for the
# EEG channels, stored as a projector (see :ref:`section-avg-ref-proj` in the
# :ref:`tut-set-eeg-ref` tutorial for more info about when you may want to do
# this). We'll discuss how to do each of these below.
#
# Since this is a combined EEG+MEG dataset, let's start by restricting the data
# to just the EEG and EOG channels. This will cause the other projectors saved
# in the file (which apply only to magnetometer channels) to be removed. By
# looking at the measurement info we can see that we now have 59 EEG channels
# and 1 EOG channel.

raw.pick(['eeg', 'eog']).load_data()
raw.info

# %%
# Channel names and types
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# In practice it's quite common to have some channels labelled as EEG that are
# actually EOG channels. `~mne.io.Raw` objects have a
# `~mne.io.Raw.set_channel_types` method that you can use to change a channel
# that is labeled as ``eeg`` into an ``eog`` type. You can also rename channels
# using the `~mne.io.Raw.rename_channels` method. Detailed examples of both of
# these methods can be found in the tutorial :ref:`tut-raw-class`. In this data
# the channel types are all correct already, so for now we'll just rename the
# channels to remove a space and a leading zero in the channel names, and
# convert to lowercase:

channel_renaming_dict = {name: name.replace(' 0', '').lower()
                         for name in raw.ch_names}
_ = raw.rename_channels(channel_renaming_dict)  # happens in-place

# %%
# Channel locations
# ^^^^^^^^^^^^^^^^^
#
# The tutorial :ref:`tut-sensor-locations` describes MNE-Python's handling of
# sensor positions in great detail. To briefly summarize: MNE-Python
# distinguishes :term:`montages <montage>` (which contain sensor positions in
# 3D: ``x``, ``y``, ``z``, in meters) from :term:`layouts <layout>` (which
# define 2D arrangements of sensors for plotting approximate overhead diagrams
# of sensor positions). Additionally, montages may specify *idealized* sensor
# positions (based on, e.g., an idealized spherical headshape model) or they
# may contain *realistic* sensor positions obtained by digitizing the 3D
# locations of the sensors when placed on the actual subject's head.
#
# This dataset has realistic digitized 3D sensor locations saved as part of the
# ``.fif`` file, so we can view the sensor locations in 2D or 3D using the
# `~mne.io.Raw.plot_sensors` method:

raw.plot_sensors(show_names=True)
fig = raw.plot_sensors('3d')

# %%
# If you're working with a standard montage like the `10-20 <ten_twenty_>`_
# system, you can add sensor locations to the data like this:
# ``raw.set_montage('standard_1020')``.  See :ref:`tut-sensor-locations` for
# info on what other standard montages are built-in to MNE-Python.
#
# If you have digitized realistic sensor locations, there are dedicated
# functions for loading those digitization files into MNE-Python; see
# :ref:`reading-dig-montages` for discussion and :ref:`dig-formats` for a list
# of supported formats. Once loaded, the digitized sensor locations can be
# added to the data by passing the loaded montage object to
# ``raw.set_montage()``.
#
#
# Setting the EEG reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As mentioned above, this data already has an EEG common average reference
# added as a :term:`projector`. We can view the effect of this on the raw data
# by plotting with and without the projector applied:

for proj in (False, True):
    fig = raw.plot(n_channels=5, proj=proj, scalings=dict(eeg=50e-6))
    fig.subplots_adjust(top=0.9)  # make room for title
    ref = 'Average' if proj else 'No'
    fig.suptitle(f'{ref} reference', size='xx-large', weight='bold')

# %%
# The referencing scheme can be changed with the function
# `mne.set_eeg_reference` (which by default operates on a *copy* of the data)
# or the `raw.set_eeg_reference() <mne.io.Raw.set_eeg_reference>` method (which
# always modifies the data in-place). The tutorial :ref:`tut-set-eeg-ref` shows
# several examples of this.
#
#
# Filtering
# ^^^^^^^^^
#
# MNE-Python has extensive support for different ways of filtering data. For a
# general discussion of filter characteristics and MNE-Python defaults, see
# :ref:`disc-filtering`. For practical examples of how to apply filters to your
# data, see :ref:`tut-filter-resample`. Here, we'll apply a simple high-pass
# filter for illustration:

raw.filter(l_freq=0.1, h_freq=None)

# %%
# Evoked responses: epoching and averaging
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The general process for extracting evoked responses from continuous data is
# to use the `~mne.Epochs` constructor, and then average the resulting epochs
# to create an `~mne.Evoked` object. In MNE-Python, events are represented as
# a :class:`NumPy array <numpy.ndarray>` of sample numbers and integer event
# codes. The event codes are stored in the last column of the events array:

np.unique(events[:, -1])

# %%
# The :ref:`tut-event-arrays` tutorial discusses event arrays in more detail.
# Integer event codes are mapped to more descriptive text using a Python
# :class:`dictionary <dict>` usually called ``event_id``. This mapping is
# determined by your experiment code (i.e., it reflects which event codes you
# chose to use to represent different experimental events or conditions). For
# the :ref:`sample-dataset` data has the following mapping:

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}

# %%
# Now we can extract epochs from the continuous data. An interactive plot
# allows you to click on epochs to mark them as "bad" and drop them from the
# analysis (it is not interactive on the documentation website, but will be
# when you run `epochs.plot() <mne.Epochs.plot>` in a Python console).

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7,
                    preload=True)
fig = epochs.plot()

# %%
# It is also possible to automatically drop epochs, when first creating them or
# later on, by providing maximum peak-to-peak signal value thresholds (pass to
# the `~mne.Epochs` constructor as the ``reject`` parameter; see
# :ref:`tut-reject-epochs-section` for details).  You can also do this after
# the epochs are already created, using the `~mne.Epochs.drop_bad` method:

reject_criteria = dict(eeg=100e-6,  # 100 µV
                       eog=200e-6)  # 200 µV
_ = epochs.drop_bad(reject=reject_criteria)

# %%
# Next we generate a barplot of which channels contributed most to epochs
# getting rejected. If one channel is responsible for lots of epoch rejections,
# it may be worthwhile to mark that channel as "bad" in the `~mne.io.Raw`
# object and then re-run epoching (fewer channels w/ more good epochs may be
# preferable to keeping all channels but losing many epochs). See
# :ref:`tut-bad-channels` for more info.

epochs.plot_drop_log()

# %%
# Another way in which epochs can be automatically dropped is if the
# `~mne.io.Raw` object they're extracted from contains :term:`annotations` that
# begin with either ``bad`` or ``edge`` ("edge" annotations are automatically
# inserted when concatenating two separate `~mne.io.Raw` objects together). See
# :ref:`tut-reject-data-spans` for more information about annotation-based
# epoch rejection.
#
# Now that we've dropped the bad epochs, let's look at our evoked responses for
# some conditions we care about. Here the `~mne.Epochs.average` method will
# create and `~mne.Evoked` object, which we can then plot. Notice that we\
# select which condition we want to average using the square-bracket indexing
# (like a :class:`dictionary <dict>`); that returns a smaller epochs object
# containing just the epochs from that condition, to which we then apply the
# `~mne.Epochs.average` method:

l_aud = epochs['auditory/left'].average()
l_vis = epochs['visual/left'].average()

# %%
# These `~mne.Evoked` objects have their own interactive plotting method
# (though again, it won't be interactive on the documentation website):
# click-dragging a span of time will generate a scalp field topography for that
# time span. Here we also demonstrate built-in color-coding the channel traces
# by location:

fig1 = l_aud.plot()
fig2 = l_vis.plot(spatial_colors=True)

# %%
# Scalp topographies can also be obtained non-interactively with the
# `~mne.Evoked.plot_topomap` method. Here we display topomaps of the average
# field in 50 ms time windows centered at -200 ms, 100 ms, and 400 ms.

l_aud.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)

# %%
# Considerable customization of these plots is possible, see the docstring of
# `~mne.Evoked.plot_topomap` for details.
#
# There is also a built-in method for combining "butterfly" plots of the
# signals with scalp topographies, called `~mne.Evoked.plot_joint`. Like
# `~mne.Evoked.plot_topomap` you can specify times for the scalp topographies
# or you can let the method choose times automatically, as is done here:

l_aud.plot_joint()

# %%
# Global field power (GFP)
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Global field power :footcite:`Lehmann1980,Lehmann1984,Murray2008` is,
# generally speaking, a measure of agreement of the signals picked up by all
# sensors across the entire scalp: if all sensors have the same value at a
# given time point, the GFP will be zero at that time point; if the signals
# differ, the GFP will be non-zero at that time point. GFP
# peaks may reflect "interesting" brain activity, warranting further
# investigation. Mathematically, the GFP is the population standard
# deviation across all sensors, calculated separately for every time point.
#
# You can plot the GFP using `evoked.plot(gfp=True) <mne.Evoked.plot>`. The GFP
# trace will be black if ``spatial_colors=True`` and green otherwise. The EEG
# reference does not affect the GFP:

# sphinx_gallery_thumbnail_number=11
for evk in (l_aud, l_vis):
    evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

# %%
# To plot the GFP by itself you can pass ``gfp='only'`` (this makes it easier
# to read off the GFP data values, because the scale is aligned):

l_aud.plot(gfp='only')

# %%
# As stated above, the GFP is the population standard deviation of the signal
# across channels. To compute it manually, we can leverage the fact that
# `evoked.data <mne.Evoked.data>` is a :class:`NumPy array <numpy.ndarray>`,
# and verify by plotting it using matplotlib commands:

gfp = l_aud.data.std(axis=0, ddof=0)

# Reproducing the MNE-Python plot style seen above
fig, ax = plt.subplots()
ax.plot(l_aud.times, gfp * 1e6, color='lime')
ax.fill_between(l_aud.times, gfp * 1e6, color='lime', alpha=0.2)
ax.set(xlabel='Time (s)', ylabel='GFP (µV)', title='EEG')

# %%
# Analyzing regions of interest (ROIs): averaging across channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Since our sample data is responses to left and right auditory and visual
# stimuli, we may want to compare left versus right ROIs. To average across
# channels in a region of interest, we first find the channel indices we want.
# Looking back at the 2D sensor plot above, we might choose the following for
# left and right ROIs:

left = ['eeg17', 'eeg18', 'eeg25', 'eeg26']
right = ['eeg23', 'eeg24', 'eeg34', 'eeg35']

left_ix = mne.pick_channels(l_aud.info['ch_names'], include=left)
right_ix = mne.pick_channels(l_aud.info['ch_names'], include=right)

# %%
# Now we can create a new Evoked with 2 virtual channels (one for each ROI):
roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
roi_evoked = mne.channels.combine_channels(l_aud, roi_dict, method='mean')
print(roi_evoked.info['ch_names'])
roi_evoked.plot()

# %%
# Comparing conditions
# ^^^^^^^^^^^^^^^^^^^^
#
# If we wanted to compare our auditory and visual stimuli, a useful function is
# `mne.viz.plot_compare_evokeds`. By default this will combine all channels in
# each evoked object using global field power (or RMS for MEG channels); here
# instead we specify to combine by averaging, and restrict it to a subset of
# channels by passing ``picks``:

evokeds = dict(auditory=l_aud, visual=l_vis)
picks = [f'eeg{n}' for n in range(10, 15)]
mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine='mean')

# %%
# We can also easily get confidence intervals by treating each epoch as a
# separate observation using the `~mne.Epochs.iter_evoked` method. A confidence
# interval across subjects could also be obtained, by passing a list of
# `~mne.Evoked` objects (one per subject) to the
# `~mne.viz.plot_compare_evokeds` function.

evokeds = dict(auditory=list(epochs['auditory/left'].iter_evoked()),
               visual=list(epochs['visual/left'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=picks)

# %%
# We can also compare conditions by subtracting one `~mne.Evoked` object from
# another using the `mne.combine_evoked` function (this function also allows
# pooling of epochs without subtraction).

aud_minus_vis = mne.combine_evoked([l_aud, l_vis], weights=[1, -1])
aud_minus_vis.plot_joint()

# %%
# .. warning::
#
#     The code above yields an **equal-weighted difference**. If you have
#     imbalanced trial numbers, you might want to equalize the number of events
#     per condition first by using `epochs.equalize_event_counts()
#     <mne.Epochs.equalize_event_counts>` before averaging.
#
#
# Grand averages
# ^^^^^^^^^^^^^^
#
# To compute grand averages across conditions (or subjects), you can pass a
# list of `~mne.Evoked` objects to `mne.grand_average`. The result is another
# `~mne.Evoked` object.

grand_average = mne.grand_average([l_aud, l_vis])
print(grand_average)

# %%
# For combining *conditions* it is also possible to make use of :term:`HED`
# tags in the condition names when selecting which epochs to average. For
# example, we have the condition names:

list(event_dict)

# %%
# We can select the auditory conditions (left and right together) by passing:

epochs['auditory'].average()

# %%
# see :ref:`tut-section-subselect-epochs` for details.
#
# The tutorials :ref:`tut-epochs-class` and :ref:`tut-evoked-class` have many
# more details about working with the `~mne.Epochs` and `~mne.Evoked` classes.

# %%
# Amplitude and Latency Measures
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is common in ERP research to extract measures of amplitude or latency to
# compare across different conditions. There are many measures that can be
# extracted from ERPs, and many of these are detailed (including the respective
# strengths and weaknesses) in Ch. 9 of Luck :footcite:`Luck2014` (also see
# the `Measurement Tool <https://bit.ly/37uydRw>`_ in the ERPLAB Toolbox
# :footcite:`Lopez-CalderonLuck2014`).
#
# This part of the tutorial will demonstrate how to extract three common
# measures:
#
# * Peak Latency
# * Peak Amplitude
# * Mean Amplitude
#
# Peak Latency and Amplitude
# --------------------------
#
# Probably most common measures of amplitude and latency are peak measures.
# Peak measures are basically the maximum amplitude of the signal in a
# specified time window, and the time point (or latency) at which the peak
# amplitude occurred.
#
# Peak measures can be obtained using the `~mne.Evoked.get_peak` method. There
<<<<<<< HEAD
# are two important things to point out about `~mne.Evoked.get_peak` method.
# First, it returns the peak latency and amplitude from **all channels** in
# the `~mne.Evoked` object.
# Second, the `~mne.Evoked.get_peak` method can find different 'types' of
=======
# are two important things to point out:
# First, it finds the strongest peak looking across **all channels** of
# the selected type that are available in the `~mne.Evoked` object. As a
# consequence if you want to restrict the search for the peak to a group of
# channels, you should first use `~mne.Evoked.pick_channels`.
# Second, the `~mne.Evoked.get_peak` method can find different types of
>>>>>>> c41898baf37a02b87c224fb12ef198293de6ad67
# peaks using the ``mode`` argument. There are three options:
#
# * ``mode='pos'``: finds the peak with a positive voltage (ignores
#   negative voltages)
# * ``mode='neg'``: finds the peak with a negative voltage (ignores
#   positive voltages)
# * ``mode='abs'``: finds the peak with the largest absolute voltage
#   regardless of sign (positive or negative)
#
# The following example demonstrates how to find positive peak in the ERP for
# the left visual condition (i.e., the ``l_vis`` `~mne.Evoked` object). The
# time window used to search for the peak is between .065 to .115 sec, and all
# ``'eeg'`` channels are used.

# Get peak amplitude and latency.
tmin, tmax = .065, .115
ch, lat, amp = l_vis.get_peak(ch_type='eeg', tmin=tmin, tmax=tmax,
                              mode='pos', return_amplitude=True)

# Convert latency and amplitude to msec and microvolts
lat *= 1e3
amp *= 1e6

# Print output
print(f'Channel: {ch}')
print(f'Peak Latency: {lat:.3f} msec')
print(f'Peak Amplitude: {amp:.3f} \u00B5V')

# %%
# The output shows that the channel ``eeg55`` had the maximum positive peak in
# the chosen time window. In practice, one might want to pull out the peak for
# an *a priori* region of interest or a single electrode depending on the study
# This can be done by combining the `~mne.Evoked.pick`
# (or `~mne.Evoked.pick_channels`) methods with the `~mne.Evoked.get_peak`
# method.
#
# Here, let's assume we believe the effects of interest will occur
# at ``eeg59``.

# Get the peak and latency measure from a single ROI
# Fist, return a copy of l_vis to select the channel from
l_vis_roi = l_vis.copy().pick('eeg59')
_, lat_roi, amp_roi = l_vis_roi.get_peak(tmin=tmin, tmax=tmax, mode='pos',
                                         return_amplitude=True)

# Convert latency and amplitude to msec and microvolts
lat_roi *= 1e3
amp_roi *= 1e6

# Print output
print('Channel: eeg59')
print(f'Peak Latency: {lat_roi:.3f} msec')
print(f'Peak Amplitude: {amp_roi:.3f} \u00B5V')

# %%
# While the peak latency is the same in channels ``eeg55`` and ``eeg59``, the
# peak amplitudes differ. The above approach can be done on virtual channels
# created with the `~mne.channels.combine_channels` function and on difference
# waves created with the `mne.combine_evoked` function (``aud_minus_vis``).
#
# While beyond the scope of this tutorial, peak measures are very susceptible
# to high frequency noise (for discussion, see :footcite:`Luck2014`). One way
# to avoid this is to apply a non-causal low-pass filters to the ERP. While
# this can reduce bias in peak amplitude measures due to high frequency
# noise, it can introduce challenges in interpreting latency measures for
# effects of interest :footcite:`Rousselet2012,VanRullen2011`.
#
# If using peak measures, it is critical to visually inspect the data to
<<<<<<< HEAD
# make sure the selected time window actually contains a peak. Note that
# `~mne.Evoked.get_peak` will always identify a peak amplitude in the time
# window being searched. However, the peak that is identified may be incorrect.
# Instead of a peak, we could just measure the rising edge of a peak, for
# instance, which is not ideal. The following demonstrates why visual
=======
# make sure the selected time window actually contains a peak
# (`~mne.Evoked.get_peak` will always identify a peak).
# Visual inspection allows to easily verify whether the automatically found
# peak is correct. The automatic procedure can identify the rising slope of
# the signal as a peak, for example, which is incorrect.
# The following example demonstrates why visual
>>>>>>> c41898baf37a02b87c224fb12ef198293de6ad67
# inspection is crucial. Below, we use a known bad time window (.09 to .12
# seconds) to search for a peak on ``eeg59``.

# Get BAD peak measures
bad_tmin, bad_tmax = .09, .12
_, bad_lat_roi, bad_amp_roi = \
    l_vis_roi.get_peak(mode='pos', tmin=bad_tmin, tmax=bad_tmax,
                       return_amplitude=True)

# Convert latency and amplitude to msec and microvolts
bad_lat_roi *= 1e3
bad_amp_roi *= 1e6

# Print output
print('** PEAK MEASURES FROM A BAD TIME WINDOW **')
print('Channel: eeg59')
print(f'Peak Latency: {bad_lat_roi:.3f} msec')
print(f'Peak Amplitude: {bad_amp_roi:.3f} \u00B5V')

# %%
# If all we had were the above values, it would be unclear if they are truly
# identifying peak or just a the rising edge. However, it becomes clear that
# the .09 to .12 second time window use for the search is missing the peak
# on ``eeg59``. This is shown in the bottom panel where we see the bad time
# window (highlighted in orange) misses the peak (the pink star). In contrast,
# the time window defined initially (.065 to .115 seconds; highlighted in blue)
# returns the actual peak instead of a value on the rising edge. Visual
# inspection will always help you to convince yourself the data returned are
# actual peaks.

# Make an empty figure handle and axis
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

# Plot the ERP, actual peak, and the good time window searched
l_vis_roi.plot(axes=ax1, time_unit='ms', show=False,
               titles='Bad time window missing peak')
ax1.plot(lat_roi, amp_roi, marker="*", color='C6')
ax1.axvspan(bad_tmin * 1e3, bad_tmax * 1e3, facecolor='C1',
            alpha=.3)
ax1.set_xlim(-50, 150)  # Show zoomed in around peak

# Plot the ERP, actual peak, and the bad time window searched
l_vis_roi.plot(axes=ax2, time_unit='ms', show=False,
               titles='Good time window finding peak')
ax2.plot(lat_roi, amp_roi, marker="*", color='C6')
ax2.axvspan(tmin * 1e3, tmax * 1e3, facecolor='C0',
            alpha=.3)
ax2.set_xlim(-50, 150)  # Show zoomed in around peak
plt.tight_layout()

# %%
# Mean Amplitude
# --------------
#
# Another common practice in ERP studies is to define a component (or effect)
# as the mean amplitude within a specified time window. One advantage of this
# approach is that it is less sensitive to high frequency noise (compared to
# peak amplitude measures) because averaging over a time window is, in essence,
# a filter.
#
# When using mean amplitude measures, it is a good idea to have a predefined
# time window for extracting mean amplitude. Selecting the time window based
# on the observed data (e.g., the grand average) can inflate false positives in
# ERP research :footcite:`LuckGaspelin2017`.
#
# Below, demonstrates how to pull out the mean amplitude between .065 sec and
# .115 sec. Note that this time window was chosen based on inspecting this
# data, which is a bad way to select a time window as just discussed. It is
# done here out of convenience and to simply demonstrate how to extract mean
# amplitude.
#
# The following code also demonstrates how to extract this for all channels
# and store the output in a pandas dataframe.

# Extract mean amplitude from eeg59 using the l_vis_roi Evoked object
l_vis_roi_cropped = l_vis_roi.copy().crop(tmin=tmin, tmax=tmax)
m_amp_roi = l_vis_roi_cropped.data.mean() * 1e6
print('Channel: eeg59')
print(f'Time Window: {tmin}s - {tmax}s')
print(f'Mean Amplitude: {m_amp_roi:.3f} \u00B5V')

# Extract mean amplitude for all channels in l_vis (including `eog`)
l_vis_cropped = l_vis.copy().crop(tmin=tmin, tmax=tmax)
m_amp_all = l_vis_cropped.data.mean(axis=1) * 1e6
n_chans = len(l_vis_cropped.info['ch_names'])
data_dict = dict(ch_name=l_vis_cropped.info['ch_names'],
                 mean_amp=m_amp_all)
m_amp_df = pd.DataFrame(data_dict)
m_amp_df['tmin'] = tmin
m_amp_df['tmax'] = tmax
m_amp_df['condition'] = 'Left/Visual'
m_amp_df.head()

# %%
# .. _ten_twenty: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
#
#
# References
# ----------
# .. footbibliography::
