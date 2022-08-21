# -*- coding: utf-8 -*-
"""
.. _tut-creating-data-structures:

================================================
Creating MNE-Python data structures from scratch
================================================

This tutorial shows how to create MNE-Python's core data structures using an
existing :class:`NumPy array <numpy.ndarray>` of (real or synthetic) data.

We begin by importing the necessary Python modules:
"""

# %%

import numpy as np

import mne

# %%
# Creating `~mne.Info` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. admonition:: Info objects
#     :class: sidebar note
#
#     For full documentation on the `~mne.Info` object, see
#     :ref:`tut-info-class`.
#
# The core data structures for continuous (`~mne.io.Raw`), discontinuous
# (`~mne.Epochs`), and averaged (`~mne.Evoked`) data all have an ``info``
# attribute comprising an `mne.Info` object. When reading recorded data using
# one of the functions in the ``mne.io`` submodule, `~mne.Info` objects are
# created and populated automatically. But if we want to create a
# `~mne.io.Raw`, `~mne.Epochs`, or `~mne.Evoked` object from scratch, we need
# to create an appropriate `~mne.Info` object as well. The easiest way to do
# this is with the `mne.create_info` function to initialize the required info
# fields. Additional fields can be assigned later as one would with a regular
# :class:`dictionary <dict>`.
#
# To initialize a minimal `~mne.Info` object requires a list of channel names,
# and the sampling frequency. As a convenience for simulated data, channel
# names can be provided as a single integer, and the names will be
# automatically created as sequential integers (starting with ``0``):

# Create some dummy metadata
n_channels = 32
sampling_freq = 200  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)
print(info)

# %%
# You can see in the output above that, by default, the channels are assigned
# as type "misc" (where it says ``chs: 32 MISC``). You can assign the channel
# type when initializing the `~mne.Info` object if you want:

ch_names = [f'MEG{n:03}' for n in range(1, 10)] + ['EOG001']
ch_types = ['mag', 'grad', 'grad'] * 3 + ['eog']
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
print(info)

# %%
# If the channel names follow one of the standard montage naming schemes, their
# spatial locations can be automatically added using the
# `~mne.Info.set_montage` method:

ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
ch_types = ['eeg'] * 7
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')

# %%
# .. admonition:: Info consistency
#     :class: sidebar warning
#
#     When assigning new values to the fields of an `~mne.Info` object, it is
#     important that the fields stay consistent. if there are ``N`` channels:
#
#     - The length of the channel information field ``chs`` must be ``N``.
#     - The length of the ``ch_names`` field must be ``N``.
#     - The ``ch_names`` field should be consistent with the ``name``
#       field of the channel information contained in ``chs``.
#
# Note the new field ``dig`` that includes our seven channel locations as well
# as theoretical values for the three
# :term:`cardinal scalp landmarks <fiducial point>`.
#
# Additional fields can be added in the same way that Python dictionaries are
# modified, using square-bracket key assignment:

info['description'] = 'My custom dataset'
info['bads'] = ['O1']  # Names of bad channels
print(info)

# %%
# Creating `~mne.io.Raw` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. admonition:: Units
#     :class: sidebar note
#
#     The expected units for the different channel types are:
#
#     - Volts: eeg, eog, seeg, dbs, emg, ecg, bio, ecog
#     - Teslas: mag
#     - Teslas/meter: grad
#     - Molar: hbo, hbr
#     - Amperes: dipole
#     - Arbitrary units: misc
#
# To create a `~mne.io.Raw` object from scratch, you can use the
# `mne.io.RawArray` class constructor, which takes an `~mne.Info` object and a
# :class:`NumPy array <numpy.ndarray>` of shape ``(n_channels, n_samples)``.
# Here, we'll create some sinusoidal data and plot it:

times = np.linspace(0, 1, sampling_freq, endpoint=False)
sine = np.sin(20 * np.pi * times)
cosine = np.cos(10 * np.pi * times)
data = np.array([sine, cosine])

info = mne.create_info(ch_names=['10 Hz sine', '5 Hz cosine'],
                       ch_types=['misc'] * 2,
                       sfreq=sampling_freq)

simulated_raw = mne.io.RawArray(data, info)
simulated_raw.plot(show_scrollbars=False, show_scalebars=False)


# %%
# Creating `~mne.Epochs` objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To create an `~mne.Epochs` object from scratch, you can use the
# `mne.EpochsArray` class constructor, which takes an `~mne.Info` object and a
# :class:`NumPy array <numpy.ndarray>` of shape ``(n_epochs, n_channels,
# n_samples)``. Here we'll create 5 epochs of our 2-channel data, and plot it.
# Notice that we have to pass ``picks='misc'`` to the `~mne.Epochs.plot`
# method, because by default it only plots :term:`data channels`.

data = np.array([[0.2 * sine, 1.0 * cosine],
                 [0.4 * sine, 0.8 * cosine],
                 [0.6 * sine, 0.6 * cosine],
                 [0.8 * sine, 0.4 * cosine],
                 [1.0 * sine, 0.2 * cosine]])

simulated_epochs = mne.EpochsArray(data, info)
simulated_epochs.plot(picks='misc', show_scrollbars=False)

# %%
# Since we did not supply an events array, the `~mne.EpochsArray` constructor
# automatically created one for us, with all epochs having the same event
# number:

print(simulated_epochs.events[:, -1])

# %%
# If we want to simulate having different experimental conditions, we can pass
# an event array (and an event ID dictionary) to the constructor. Since our
# epochs are 1 second long and have 200 samples/second, we'll put our events
# spaced 200 samples apart, and pass ``tmin=-0.5``, so that the events
# land in the middle of each epoch (the events are always placed at time=0 in
# each epoch).

events = np.column_stack((np.arange(0, 1000, sampling_freq),
                          np.zeros(5, dtype=int),
                          np.array([1, 2, 1, 2, 1])))
event_dict = dict(condition_A=1, condition_B=2)
simulated_epochs = mne.EpochsArray(data, info, tmin=-0.5, events=events,
                                   event_id=event_dict)
simulated_epochs.plot(picks='misc', show_scrollbars=False, events=events,
                      event_id=event_dict)

# %%
# You could also create simulated epochs by using the normal `~mne.Epochs`
# (not `~mne.EpochsArray`) constructor on the simulated `~mne.io.RawArray`
# object, by creating an events array (e.g., using
# `mne.make_fixed_length_events`) and extracting epochs around those events.
#
#
# Creating `~mne.Evoked` Objects
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you already have data that was averaged across trials, you can use it to
# create an `~mne.Evoked` object using the `~mne.EvokedArray` class
# constructor.  It requires an `~mne.Info` object and a data array of shape
# ``(n_channels, n_times)``, and has an optional ``tmin`` parameter like
# `~mne.EpochsArray` does. It also has a parameter ``nave`` indicating how many
# trials were averaged together, and a ``comment`` parameter useful for keeping
# track of experimental conditions, etc. Here we'll do the averaging on our
# NumPy array and use the resulting averaged data to make our `~mne.Evoked`.

# Create the Evoked object
evoked_array = mne.EvokedArray(data.mean(axis=0), info, tmin=-0.5,
                               nave=data.shape[0], comment='simulated')
print(evoked_array)
evoked_array.plot()
