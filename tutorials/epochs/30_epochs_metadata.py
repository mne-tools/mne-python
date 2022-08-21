# -*- coding: utf-8 -*-
"""
.. _tut-epochs-metadata:

===========================
Working with Epoch metadata
===========================

This tutorial shows how to add metadata to `~mne.Epochs` objects, and
how to use :ref:`Pandas query strings <pandas:indexing.query>` to select and
plot epochs based on metadata properties.

For this tutorial we'll use a different dataset than usual: the
:ref:`kiloword-dataset`, which contains EEG data averaged across 75 subjects
who were performing a lexical decision (word/non-word) task. The data is in
`~mne.Epochs` format, with each epoch representing the response to a
different stimulus (word). As usual we'll start by importing the modules we
need and loading the data:
"""

# %%

import numpy as np
import pandas as pd
import mne

kiloword_data_folder = mne.datasets.kiloword.data_path()
kiloword_data_file = kiloword_data_folder / 'kword_metadata-epo.fif'
epochs = mne.read_epochs(kiloword_data_file)

# %%
# Viewing ``Epochs`` metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. admonition:: Restrictions on metadata DataFrames
#    :class: sidebar warning
#
#    Metadata dataframes are less flexible than typical
#    :class:`Pandas DataFrames <pandas.DataFrame>`. For example, the allowed
#    data types are restricted to strings, floats, integers, or booleans;
#    and the row labels are always integers corresponding to epoch numbers.
#    Other capabilities of :class:`DataFrames <pandas.DataFrame>` such as
#    :class:`hierarchical indexing <pandas.MultiIndex>` are possible while the
#    `~mne.Epochs` object is in memory, but will not survive saving and
#    reloading the `~mne.Epochs` object to/from disk.
#
# The metadata attached to `~mne.Epochs` objects is stored as a
# :class:`pandas.DataFrame` containing one row for each epoch. The columns of
# this :class:`~pandas.DataFrame` can contain just about any information you
# want to store about each epoch; in this case, the metadata encodes
# information about the stimulus seen on each trial, including properties of
# the visual word form itself (e.g., ``NumberOfLetters``, ``VisualComplexity``)
# as well as properties of what the word means (e.g., its ``Concreteness``) and
# its prominence in the English lexicon (e.g., ``WordFrequency``). Here are all
# the variables; note that in a Jupyter notebook, viewing a
# :class:`pandas.DataFrame` gets rendered as an HTML table instead of the
# normal Python output block:

epochs.metadata

# %%
# Viewing the metadata values for a given epoch and metadata variable is done
# using any of the :ref:`Pandas indexing <pandas:/reference/indexing.rst>`
# methods such as :obj:`~pandas.DataFrame.loc`,
# :obj:`~pandas.DataFrame.iloc`, :obj:`~pandas.DataFrame.at`,
# and :obj:`~pandas.DataFrame.iat`. Because the
# index of the dataframe is the integer epoch number, the name- and index-based
# selection methods will work similarly for selecting rows, except that
# name-based selection (with :obj:`~pandas.DataFrame.loc`) is inclusive of the
# endpoint:

print('Name-based selection with .loc')
print(epochs.metadata.loc[2:4])

print('\nIndex-based selection with .iloc')
print(epochs.metadata.iloc[2:4])

# %%
# Modifying the metadata
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Like any :class:`pandas.DataFrame`, you can modify the data or add columns as
# needed. Here we convert the ``NumberOfLetters`` column from :class:`float` to
# :class:`integer <int>` data type, and add a :class:`boolean <bool>` column
# that arbitrarily divides the variable ``VisualComplexity`` into high and low
# groups.

epochs.metadata['NumberOfLetters'] = \
    epochs.metadata['NumberOfLetters'].map(int)

epochs.metadata['HighComplexity'] = epochs.metadata['VisualComplexity'] > 65
epochs.metadata.head()

# %%
# Selecting epochs using metadata queries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# All `~mne.Epochs` objects can be subselected by event name, index, or
# :term:`slice` (see :ref:`tut-section-subselect-epochs`). But
# `~mne.Epochs` objects with metadata can also be queried using
# :ref:`Pandas query strings <pandas:indexing.query>` by passing the query
# string just as you would normally pass an event name. For example:

print(epochs['WORD.str.startswith("dis")'])

# %%
# This capability uses the :meth:`pandas.DataFrame.query` method under the
# hood, so you can check out the documentation of that method to learn how to
# format query strings. Here's another example:

print(epochs['Concreteness > 6 and WordFrequency < 1'])

# %%
# Note also that traditional epochs subselection by condition name still works;
# MNE-Python will try the traditional method first before falling back on rich
# metadata querying.

epochs['solenoid'].plot_psd()

# %%
# One use of the Pandas query string approach is to select specific words for
# plotting:

words = ['typhoon', 'bungalow', 'colossus', 'drudgery', 'linguist', 'solenoid']
epochs['WORD in {}'.format(words)].plot(n_channels=29)

# %%
# Notice that in this dataset, each "condition" (A.K.A., each word) occurs only
# once, whereas with the :ref:`sample-dataset` dataset each condition (e.g.,
# "auditory/left", "visual/right", etc) occurred dozens of times. This makes
# the Pandas querying methods especially useful when you want to aggregate
# epochs that have different condition names but that share similar stimulus
# properties. For example, here we group epochs based on the number of letters
# in the stimulus word, and compare the average signal at electrode ``Pz`` for
# each group:

evokeds = dict()
query = 'NumberOfLetters == {}'
for n_letters in epochs.metadata['NumberOfLetters'].unique():
    evokeds[str(n_letters)] = epochs[query.format(n_letters)].average()

# sphinx_gallery_thumbnail_number = 3
mne.viz.plot_compare_evokeds(evokeds, cmap=('word length', 'viridis'),
                             picks='Pz')

# %%
# Metadata can also be useful for sorting the epochs in an image plot. For
# example, here we order the epochs based on word frequency to see if there's a
# pattern to the latency or intensity of the response:

sort_order = np.argsort(epochs.metadata['WordFrequency'])
epochs.plot_image(order=sort_order, picks='Pz')

# %%
# Although there's no obvious relationship in this case, such analyses may be
# useful for metadata variables that more directly index the time course of
# stimulus processing (such as reaction time).
#
#
# Adding metadata to an ``Epochs`` object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can add a metadata :class:`~pandas.DataFrame` to any
# `~mne.Epochs` object (or replace existing metadata) simply by
# assigning to the :attr:`~mne.Epochs.metadata` attribute:

new_metadata = pd.DataFrame(data=['foo'] * len(epochs), columns=['bar'],
                            index=range(len(epochs)))
epochs.metadata = new_metadata
epochs.metadata.head()

# %%
# You can remove metadata from an `~mne.Epochs` object by setting its
# metadata to ``None``:

epochs.metadata = None
