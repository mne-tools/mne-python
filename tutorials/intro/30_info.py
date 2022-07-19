# -*- coding: utf-8 -*-
"""
.. _tut-info-class:

=======================
The Info data structure
=======================

This tutorial describes the :class:`mne.Info` data structure, which keeps track
of various recording details, and is attached to :class:`~mne.io.Raw`,
:class:`~mne.Epochs`, and :class:`~mne.Evoked` objects.

We will begin by loading the Python modules we need, and loading the same
:ref:`example data <sample-dataset>` we used in the :ref:`introductory tutorial
<tut-overview>`:
"""

# %%

import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

# %%
# As seen in the :ref:`introductory tutorial <tut-overview>`, when a
# :class:`~mne.io.Raw` object is loaded, an :class:`~mne.Info` object is
# created automatically, and stored in the ``raw.info`` attribute:

print(raw.info)

# %%
# However, it is not strictly necessary to load the :class:`~mne.io.Raw` object
# in order to view or edit the :class:`~mne.Info` object; you can extract all
# the relevant information into a stand-alone :class:`~mne.Info` object using
# :func:`mne.io.read_info`:

info = mne.io.read_info(sample_data_raw_file)
print(info)

# %%
# As you can see, the :class:`~mne.Info` object keeps track of a lot of
# information about:
#
# - the recording system (gantry angle, HPI details, sensor digitizations,
#   channel names, ...)
# - the experiment (project name and ID, subject information, recording date,
#   experimenter name or ID, ...)
# - the data (sampling frequency, applied filter frequencies, bad channels,
#   projectors, ...)
#
# The complete list of fields is given in :class:`the API documentation
# <mne.Info>`.
#
#
# Querying the ``Info`` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The fields in a :class:`~mne.Info` object act like Python :class:`dictionary
# <dict>` keys, using square brackets and strings to access the contents of a
# field:

print(info.keys())
print()  # insert a blank line
print(info['ch_names'])

# %%
# Most of the fields contain :class:`int`, :class:`float`, or :class:`list`
# data, but the ``chs`` field bears special mention: it contains a list of
# dictionaries (one :class:`dict` per channel) containing everything there is
# to know about a channel other than the data it recorded. Normally it is not
# necessary to dig into the details of the ``chs`` field — various MNE-Python
# functions can extract the information more cleanly than iterating over the
# list of dicts yourself — but it can be helpful to know what is in there. Here
# we show the keys for the first channel's :class:`dict`:

print(info['chs'][0].keys())

# %%
# .. _picking_channels:
#
# Obtaining subsets of channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It is often useful to convert between channel names and the integer indices
# identifying rows of the data array where those channels' measurements are
# stored. The :class:`~mne.Info` object is useful for this task; two
# convenience functions that rely on the :class:`mne.Info` object for picking
# channels are :func:`mne.pick_channels` and :func:`mne.pick_types`.
# :func:`~mne.pick_channels` minimally takes a list of all channel names and a
# list of channel names to include; it is also possible to provide an empty
# list to ``include`` and specify which channels to ``exclude`` instead:

print(mne.pick_channels(info['ch_names'], include=['MEG 0312', 'EEG 005']))

print(mne.pick_channels(info['ch_names'], include=[],
                        exclude=['MEG 0312', 'EEG 005']))

# %%
# :func:`~mne.pick_types` works differently, since channel type cannot always
# be reliably determined from channel name alone. Consequently,
# :func:`~mne.pick_types` needs an :class:`~mne.Info` object instead of just a
# list of channel names, and has boolean keyword arguments for each channel
# type. Default behavior is to pick only MEG channels (and MEG reference
# channels if present) and exclude any channels already marked as "bad" in the
# ``bads`` field of the :class:`~mne.Info` object. Therefore, to get *all* and
# *only* the EEG channel indices (including the "bad" EEG channels) we must
# pass ``meg=False`` and ``exclude=[]``:

print(mne.pick_types(info, meg=False, eeg=True, exclude=[]))

# %%
# Note that the ``meg`` and ``fnirs`` parameters of :func:`~mne.pick_types`
# accept strings as well as boolean values, to allow selecting only
# magnetometer or gradiometer channels (via ``meg='mag'`` or ``meg='grad'``) or
# to pick only oxyhemoglobin or deoxyhemoglobin channels (via ``fnirs='hbo'``
# or ``fnirs='hbr'``, respectively).
#
# A third way to pick channels from an :class:`~mne.Info` object is to apply
# `regular expression`_ matching to the channel names using
# :func:`mne.pick_channels_regexp`. Here the ``^`` represents the beginning of
# the string and ``.`` character matches any single character, so both EEG and
# EOG channels will be selected:

print(mne.pick_channels_regexp(info['ch_names'], '^E.G'))

# %%
# :func:`~mne.pick_channels_regexp` can be especially useful for channels named
# according to the `10-20 <ten-twenty_>`_ system (e.g., to select all channels
# ending in "z" to get the midline, or all channels beginning with "O" to get
# the occipital channels). Note that :func:`~mne.pick_channels_regexp` uses the
# Python standard module :mod:`re` to perform regular expression matching; see
# the documentation of the :mod:`re` module for implementation details.
#
# .. warning::
#    Both :func:`~mne.pick_channels` and :func:`~mne.pick_channels_regexp`
#    operate on lists of channel names, so they are unaware of which channels
#    (if any) have been marked as "bad" in ``info['bads']``. Use caution to
#    avoid accidentally selecting bad channels.
#
#
# Obtaining channel type information
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sometimes it can be useful to know channel type based on its index in the
# data array. For this case, use :func:`mne.channel_type`, which takes
# an :class:`~mne.Info` object and a single integer channel index:

print(mne.channel_type(info, 25))

# %%
# To obtain several channel types at once, you could embed
# :func:`~mne.channel_type` in a :term:`list comprehension`, or use the
# :meth:`~mne.io.Raw.get_channel_types` method of a :class:`~mne.io.Raw`,
# :class:`~mne.Epochs`, or :class:`~mne.Evoked` instance:

picks = (25, 76, 77, 319)
print([mne.channel_type(info, x) for x in picks])
print(raw.get_channel_types(picks=picks))

# %%
# Alternatively, you can get the indices of all channels of *all* channel types
# present in the data, using :func:`~mne.channel_indices_by_type`,
# which returns a :class:`dict` with channel types as keys, and lists of
# channel indices as values:

ch_idx_by_type = mne.channel_indices_by_type(info)
print(ch_idx_by_type.keys())
print(ch_idx_by_type['eog'])

# %%
# Dropping channels from an ``Info`` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# If you want to modify an :class:`~mne.Info` object by eliminating some of the
# channels in it, you can use the :func:`mne.pick_info` function to pick the
# channels you want to keep and omit the rest:

print(info['nchan'])
eeg_indices = mne.pick_types(info, meg=False, eeg=True)
print(mne.pick_info(info, eeg_indices)['nchan'])

# %%
# We can also get a nice HTML representation in IPython like:

info

# %%
# By default, :func:`~mne.pick_info` will make a copy of the original
# :class:`~mne.Info` object before modifying it; if you want to modify it
# in-place, include the parameter ``copy=False``.
#
#
# .. LINKS
#
# .. _`regular expression`: https://en.wikipedia.org/wiki/Regular_expression
# .. _`ten-twenty`: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
