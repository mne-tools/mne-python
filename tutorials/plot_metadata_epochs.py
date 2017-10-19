"""
================================================
Pandas querying and metadata with Epochs objects
================================================

Demonstrating pandas-style string querying with Epochs metadata.
For related uses of :class:`mne.Epochs`, see the starting tutorial
:ref:`sphx_glr_auto_tutorials_plot_object_epochs.py`.

Sometimes you've got a more complex trials structure that cannot be easily
summarized as a set of unique integers. In this case, it may be useful to use
the ``metadata`` attribute of :class:`mne.Epochs` objects. This must be a
:class:`pandas.DataFrame` where each row corresponds to an epoch, and each
column corresponds to a metadata attribute of each epoch. Columns must
contain either strings, ints, or floats.

In this dataset, subjects were presented with individual words
on a screen, and the EEG activity in response to each word was recorded.
We know which word was displayed in each epoch, as well as
extra information about the word (e.g., word frequency).

Loading the data
----------------
First we'll load the data. If metadata exists for an :class:`mne.Epochs`
fif file, it will automatically be loaded in the ``metadata`` attribute.
"""

# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import mne
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the internet
path = mne.datasets.kiloword.data_path() + '/kword_metadata-epo.fif'
epochs = mne.read_epochs(path)

# The metadata exists as a Pandas DataFrame
print(epochs.metadata.head(10))

###############################################################################
# We can use this metadata attribute to select subsets of Epochs. This
# uses the Pandas :meth:`pandas.DataFrame.query` method under the hood.
# Any valid query string will work. Below we'll make two plots to compare
# between them:

av1 = epochs['Concreteness < 5 and WordFrequency < 2'].average()
av2 = epochs['Concreteness > 5 and WordFrequency > 2'].average()

av1.plot_joint(show=False)
av2.plot_joint(show=False)

###############################################################################
# Next we'll choose a subset of words to keep.
words = ['film', 'cent', 'shot', 'cold', 'main']
epochs['WORD in {}'.format(words)].plot_image(show=False)

###############################################################################
# Note that traditional epochs sub-selection still works. The traditional
# MNE methods for selecting epochs will supersede the rich metadata querying.
epochs['cent'].average().plot(show=False)

###############################################################################
# Below we'll show a more involved example that leverages the metadata
# of each epoch. We'll create a new column in our metadata object and use
# it to generate averages for many subsets of trials.

# Create a new metadata column
meta = epochs.metadata
is_concrete = meta["WordFrequency"] > meta["WordFrequency"].median()
meta["is_concrete"] = np.where(is_concrete, 'Concrete', 'Abstract')
epochs.metadata = meta

# We'll create a dictionary so that we can plot with ``plot_compare_evokeds``
categories = ["NumberOfLetters", "is_concrete"]
avs = {}
for (cat1, cat2), _ in epochs.metadata.groupby(categories):
    query = 'NumberOfLetters == {} and is_concrete == "{}"'.format(cat1, cat2)
    this_epochs = epochs[query]
    avs["{}/{}".format(cat1, cat2)] = this_epochs.average()

# Style the plot
colors = np.linspace(0, 1, num=len(avs))
style_plot = dict(
    colors=plt.get_cmap('YlGnBu_r')(colors),
    linestyles={'Concrete': '-', 'Abstract': '--'}
)

# Make the plot
ix_plot = mne.pick_channels(epochs.ch_names, ['Pz'])
fig, ax = plt.subplots(figsize=(6, 3))
fig = mne.viz.evoked.plot_compare_evokeds(
    avs, picks=ix_plot, show=False, axes=ax, **style_plot)
ax.legend(loc=[1.05, .1])
plt.show()

###############################################################################
# .. note::
#
#    Creating an :class:`mne.Epochs` object with metadata is done by passing
#    a :class:`pandas.DataFrame` to the ``metadata`` kwarg as follows:

data = epochs.get_data()
metadata = epochs.metadata.copy()
epochs_new = mne.EpochsArray(data, epochs.info, metadata=metadata)
