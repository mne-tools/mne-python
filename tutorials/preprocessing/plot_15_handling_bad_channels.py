# -*- coding: utf-8 -*-
"""
.. _tut-bad-channels:

Interpolating bad channels
==========================

This tutorial covers manual marking of bad channels and reconstructing bad
channels based on good signals at other sensors.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need, and loading some example
data:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

###############################################################################
# Marking bad channels
# ^^^^^^^^^^^^^^^^^^^^
#
# Sometimes individual channels malfunction and provide data that is too noisy
# to be usable. MNE-Python makes it easy to remove bad channels from the
# analysis stream without actually deleting the data in those channels, by
# keeping track of the bad channel indices in a list and looking at that list
# when doing analysis or plotting tasks. The list of bad channels is stored in
# the `'bads'` field of the :class:`~mne.Info` object that is attached to
# :class:`~mne.io.Raw`, :class:`~mne.Epochs`, and :class:`~mne.Evoked` objects.

print(raw.info['bads'])

###############################################################################
# Here you can see that the :file:`.fif` file we loaded from disk must have
# been keeping track of channels marked as "bad" — which is good news, because
# it means any changes we make to the list of bad channels will be preserved
# if we save our data at intermediate stages and re-load it later. In the case
# of the example data, you can see why those channels were marked bad by
# using the standard :meth:`~mne.io.Raw.plot` method; to make it easier to see
# we'll use the :func:`~mne.pick_channels_regexp` function to narrow down which
# channels we see:

# sphinx_gallery_thumbnail_number = 2
for pattern in (r'MEG 24..', r'EEG 05.'):
    picks = mne.pick_channels_regexp(raw.ch_names, regexp=pattern)
    raw.plot(order=picks, n_channels=len(picks))

###############################################################################
# Notice first of all that the channels marked as "bad" are plotted in a light
# gray color in a layer behind the other channels, to make it easy to
# distinguish them from "good" channels. The plots make it clear that `EEG 053`
# is not picking up scalp potentials at all, and `MEG 2443` looks like it's
# picking up a lot *more* than its neighbors — its signal is at least an order
# of magnitude greater than the other MEG channels. Regardless of what is
# causing the problems, neither are realistic reflections of the
# electrophysiological signals we expect, so it makes sense to exclude them.
#
# If you want to change which channels are marked as bad, you can edit
# ``raw.info['bads']`` directly; it's an ordinary Python :class:`list` so the
# usual list methods will work:

original_bads = raw.info['bads']
raw.info['bads'].append('EEG 050')               # add a single channel
raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
bad_chan = raw.info['bads'].pop(-1)  # remove the last entry in the list
raw.info['bads'] = original_bads     # change the whole list at once

###############################################################################
# .. sidebar:: Blocking execution
#
#     If you want to build an interactive bad-channel-marking step into an
#     analysis script, be sure to include the parameter ``block=True`` in your
#     call to ``raw.plot()`` or ``epochs.plot()``. This will pause the script
#     while the plot is open, giving you time to mark bad channels before
#     subsequent analysis or plotting steps are executed.
#
# You can also interactively toggle whether a channel is marked "bad" in the
# plot windows of ``raw.plot()`` or ``epochs.plot()`` by clicking on the
# channel name along the vertical axis (in ``raw.plot()`` windows you can also
# do this by clicking the channel's trace in the plot area). The ``bads`` field
# gets updated immediately each time you toggle a channel, and will retain its
# modified state after the plot window is closed.
#
# The list of bad channels in the :class:`mne.Info` object's ``bads`` field is
# automatically taken into account in dozens of functions and methods across
# the MNE-Python codebase. This is done consistently with a parameter
# ``exclude='bads'`` in the function or method signature. Typically this
# ``exclude`` parameter also accepts a list of channel names or indices, so if
# you want to *include* the bad channels you can do so by passing
# ``exclude=[]`` (or some other list of channels to exclude).
#
#
# When to look for bad channels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can start looking for bad channels during the experiment session when the
# data is being acquired. If you notice any flat or excessively noisy channels,
# you can note them in your experiment log or protocol sheet. If your system
# computes on-line averages, these can be a good way to spot bad channels as
# well. After the data has been collected, you can do a more thorough check for
# bad channels by browsing the raw data using :meth:`mne.io.Raw.plot`, with any
# projectors or ICA applied. Finally, you can compute off-line averages (again
# with projectors, ICA, and EEG referencing disabled) to look for channels with
# unusual properties.
#
# Remember, marking bad channels should be done as early as possible in the
# analysis pipeline. When bad channels are marked in a :class:`~mne.io.Raw`
# object, the markings will be automatically transferred through the chain of
# derived object types: including :class:`~mne.Epochs` and :class:`~mne.Evoked`
# objects, but also :class:`noise covariance <mne.Covariance>` objects,
# :class:`forward solution computations <mne.Forward>`, and
# :class:`inverse operators <mne.minimum_norm.InverseOperator>`.
#
#
# Why mark bad channels at all?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Many analysis computations can be strongly affected by the presence of bad
# channels. For example, a malfunctioning channel with completely flat signal
# will have zero channel variance, which will cause noise estimates to be
# unrealistically low. This low noise estimate will lead to a strong channel
# weight in the estimate of cortical current, and because the channel is flat,
# the magnitude of cortical current estimates will shrink dramatically.
#
# Conversely, very noisy channels can also cause problems. For example, they
# can lead to too many epochs being discarded based on signal amplitude
# rejection thresholds, which in turn can lead to less robust estimation of the
# noise covariance across sensors. Noisy channels can also interfere with
# :term:`SSP <projector>` computations, because the projectors will be
# spatially biased in the direction of the noisy channel, which can cause
# adjacent good channels to be suppressed.
#
#
# Interpolating bad channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In some cases simply excluding bad channels is sufficient (for example, if
# you plan only to analyze a specific sensor ROI, and the bad channel is
# outside that ROI). But it is also possible to reconstruct a bad channel by
# interpolating its signal based on the signals of the good sensors around it.
#
#
# How interpolation works
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Interpolation of EEG channels in MNE-Python is done using the spherical
# spline method [1]_, which projects the sensor locations onto a unit sphere
# and interpolates the signal at the bad sensor locations based on the signals
# at the good locations. Mathematical details are presented in
# :ref:`channel_interpolation`. Interpolation of MEG channels uses the field
# mapping algorithms used in computing the :ref:`forward solution
# <tut-forward>`.
#
#
# Interpolation in MNE-Python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Interpolating bad channels in :class:`~mne.io.Raw` objects is done with the
# :meth:`~mne.io.Raw.interpolate_bads` method, which automatically applies the
# correct method (spherical splines or field interpolation) to EEG and MEG
# channels, respectively (there is a corresponding method
# :meth:`mne.Epochs.interpolate_bads` that works for :class:`~mne.Epochs`
# objects). To illustrate how it works, we'll start by cropping the raw object
# to just three seconds for easier plotting:

raw.crop(tmin=0, tmax=3).load_data()

###############################################################################
# By default, :meth:`~mne.io.Raw.interpolate_bads` will clear out
# ``raw.info['bads']`` after interpolation, so that the interpolated channels
# are no longer excluded from subsequent computations. Here, for illustration
# purposes, we'll prevent that by specifying ``reset_bads=False`` so that when
# we plot the data before and after interpolation, the affected channels will
# still plot in red:

eeg_data = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

for data in (eeg_data, eeg_data_interp):
    data.plot(butterfly=True, color='#00000022', bad_color='r')

###############################################################################
# Note that we used the ``exclude=[]`` trick in the call to
# :meth:`~mne.io.Raw.pick_types` to make sure the bad channels were not
# automatically dropped from the selection. Here is the corresponding example
# with the interpolated gradiometer channel; since there are more channels
# we'll use a more transparent gray color this time:

grad_data = raw.copy().pick_types(meg='grad', exclude=[])
grad_data_interp = grad_data.copy().interpolate_bads(reset_bads=False)

for data in (grad_data, grad_data_interp):
    data.plot(butterfly=True, color='#00000011', bad_color='r')

###############################################################################
# References
# ^^^^^^^^^^
#
# .. [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
#        Spherical splines for scalp potential and current density mapping.
#        *Electroencephalography Clinical Neurophysiology* 72(2):184-187.
