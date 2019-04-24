# -*- coding: utf-8 -*-
"""
.. _interpolating-bads-tutorial:

Interpolating bad channels
==========================

.. include:: ../../tutorial_links.inc

This tutorial covers reconstructing bad channels based on good signals at other
sensors.
"""

###############################################################################
# As usual we'll start by importing the modules we need, and loading some
# example data:

import os
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

###############################################################################
# How interpolation works
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Interpolation of EEG channels is done using the spherical spline method [1]_,
# which projects the sensor locations onto a unit sphere and interpolates the
# signal at the bad sensor locations based on the signals at the good
# locations. Mathematical details are presented in
# :ref:`channel_interpolation`. Interpolation of MEG channels uses the field
# mapping algorithms used in computing the :ref:`forward solution
# <tut_forward>`.
#
#
# Interpolation in MNE-Python
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Interpolating bad channels in MNE-Python is done with the
# :meth:`~mne.io.Raw.interpolate_bads` method, which automatically applies the
# correct method (spherical splines or field interpolation) to EEG and MEG
# channels, respectively. We'll start by cropping the raw object to just three
# seconds for easier plotting, then create ``interp_raw`` with interpolated
# channels:

raw.crop(tmin=0, tmax=3).load_data()
interp_raw = raw.copy().interpolate_bads()

###############################################################################
# To see the effect of interpolation, we'll plot the good channels in gray and
# the bad or interpolated channels in red, separately for EEG channels and for
# gradiometers. To do that, we'll need to find the channel indices of the good
# and bad channels of each type.
#
# Since there are only two entries in bads, we can unpack them into two
# variables on one line by assigning to a tuple. This will give us channel
# *names* rather than indices, but we've already seen that we can pick channels
# by name:

print(raw.info['bads'])
bad_meg_channel, bad_eeg_channel = raw.info['bads']

bad_meg_data = raw.get_data(picks=bad_meg_channel)
bad_eeg_data = raw.get_data(picks=bad_eeg_channel)

###############################################################################
# In this dataset, good EEG channels would be easy to pick out from their
# channel names, using a list comprehension and the Python string
# ``.startswith`` method:

good_eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG') and
                     ch not in raw.info['bads']]

###############################################################################
# However, magnetometer and gradiometer channel names all start with "MEG" so
# they are not so easy to separate out by name. We've seen in
# :ref:`raw-pick-types` how to pick channels from a :class:`~mne.io.Raw` object
# with the :meth:`~mne.io.Raw.pick_types` method, but since we need to do this
# twice (once for EEG channels and once for MEG) it would necessitate making a
# copy of the :class:`~mne.io.Raw` object. To avoid copying, we can use the
# function :func:`mne.pick_types` instead, which uses the :class:`~mne.Info`
# object to pick channels, and returns an array of integer channel indices:

good_meg_indices = mne.pick_types(raw.info, meg='grad', exclude='bads')
good_eeg_indices = mne.pick_types(raw.info, meg=False, eeg=True,
                                  exclude='bads')

###############################################################################
# Now we can pass the channel indices to the :meth:`~mne.io.Raw.get_data`
# method without needing to copy the :class:`~mne.io.Raw` object. While we're
# at it, we'll get an array of times to use when plotting:

good_meg_data = raw.get_data(picks=good_meg_indices)
good_eeg_data, times = raw.get_data(picks=good_eeg_indices, return_times=True)

###############################################################################
# Now we've got the good data from EEG and magnetometer channels, and we've got
# the bad channel data *before* interpolation, so now we just need the bad
# channels *after* interpolation. We can again use the ``picks`` parameter of
# the :meth:`~mne.io.Raw.get_data` method, but this time on the ``interp_raw``
# variable:

# get the interpolated bad channels
interp_meg_data = interp_raw.get_data(picks=bad_meg_channel)
interp_eeg_data = interp_raw.get_data(picks=bad_eeg_channel)

###############################################################################
# Finally, we'll convert MEG units from teslas to femtoteslas and EEG units
# from volts to microvolts, for nicer plot axes:

# convert gradiometers to fT/mm
good_meg_data /= 1e-12
bad_meg_data /= 1e-12
interp_meg_data /= 1e-12

# convert EEG to μV
good_eeg_data /= 1e-6
bad_eeg_data /= 1e-6
interp_eeg_data /= 1e-6

###############################################################################
# To make the plots clear, we'll draw good channels in black with thin,
# semi-transparent lines and bad (or interpolated bad) channels in red with
# thick lines. To make that easier, we'll define a couple of dictionaries and
# use ``**`` `argument expansion`_ to assign the keyword args repeatedly to
# each plot:

good_kwargs = dict(color='k', linewidth=0.2, alpha=0.2)
bad_kwargs = dict(color='r', linewidth=0.4)

fig, axs = plt.subplots(2, 2, sharex=True, sharey='row')
# MEG: original (bad channel behind good, otherwise we can't see good channels)
axs[0, 0].plot(times, bad_meg_data.T, **bad_kwargs)
axs[0, 0].plot(times, good_meg_data.T, **good_kwargs)
# MEG: interpolated
axs[0, 1].plot(times, good_meg_data.T, **good_kwargs)
axs[0, 1].plot(times, interp_meg_data.T, **bad_kwargs)
# EEG: original
axs[1, 0].plot(times, good_eeg_data.T, **good_kwargs)
axs[1, 0].plot(times, bad_eeg_data.T, **bad_kwargs)
# EEG: interpolated
axs[1, 1].plot(times, good_eeg_data.T, **good_kwargs)
axs[1, 1].plot(times, interp_eeg_data.T, **bad_kwargs)

# label axes and zoom in MEG a little
axs[0, 0].set_title('Original')
axs[0, 1].set_title('Interpolated')
axs[0, 0].set_ylabel('Gradiometers (fT/mm)')
axs[1, 0].set_ylabel('EEG (μV)')
axs[1, 0].set_xlabel('time (s)')
axs[1, 1].set_xlabel('time (s)')
axs[0, 0].set_ylim(-300, 300)
fig.tight_layout()

###############################################################################
# By default, when bad channels are interpolated they are removed from
# ``raw.info['bads']``. If you want to keep them marked as "bad" after
# interpolation, :meth:`~mne.io.Raw.interpolate_bads` has a boolean
# ``reset_bads`` parameter. Setting this to ``False`` makes it easy to
# highlight interpolated channels using MNE-Python's built-in plotting methods:

(raw
 .pick_types(meg=False, eeg=True, exclude=[])
 .interpolate_bads(reset_bads=False)
 .plot(butterfly=True, color='#00000011', bad_color='r'))


###############################################################################
# References
# ^^^^^^^^^^
#
# .. [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
#        Spherical splines for scalp potential and current density mapping.
#        *Electroencephalography Clinical Neurophysiology* 72(2):184-187.
