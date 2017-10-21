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

# Create two new metadata columns
meta = epochs.metadata
is_concrete = meta["Concreteness"] > meta["Concreteness"].median()
meta["is_concrete"] = np.where(is_concrete, 'Concrete', 'Abstract')
is_concrete = meta["NumberOfLetters"] > 5
meta["is_long"] = np.where(is_concrete, 'Long', 'Short')
epochs.metadata = meta

###############################################################################
# Now we can quickly extract (and plot) subsets of the data. For example, to
# look at words split by word length and concreteness:

query = "is_long == '{0}' & is_concrete == '{1}'"
evokeds = dict()
for concreteness in ("Concrete", "Abstract"):
    for length in ("Long", "Short"):
        subset = epochs[query.format(length, concreteness)]
        evokeds["/".join((concreteness, length))] = list(subset.iter_evoked())

# For the actual visualisation, we store a number of shared parameters.
style_plot = dict(
    colors={"Long": "Crimson", "Short": "Cornflowerblue"},
    linestyles={"Concrete": "-", "Abstract": ":"},
    split_legend=True,
    ci=.68,
    show_sensors=4,
    show_legend=3,
    truncate_yaxis="max_ticks",
    picks=epochs.ch_names.index("Pz"),
)

fig, ax = plt.subplots(figsize=(6, 4))
mne.viz.plot_compare_evokeds(evokeds, axes=ax, **style_plot)
plt.show()

###############################################################################
# To compare words which are 4, 5, 6, 7 or 8 letters long:

evokeds = dict()
for nlet in epochs.metadata["NumberOfLetters"].unique():
    evokeds[str(nlet)] = epochs["NumberOfLetters == " + str(nlet)].average()

style_plot["colors"] = {str(nlet): int(nlet) for nlet in
                        epochs.metadata["NumberOfLetters"].unique()}
style_plot["cmap"] = "summer_r"
del style_plot['linestyles']

fig, ax = plt.subplots(figsize=(6, 4))
mne.viz.plot_compare_evokeds(evokeds, axes=ax, **style_plot)
plt.show()

###############################################################################
# And finally, for the interaction between concreteness and continuous length
# in letters:
evokeds = dict()
query = "is_concrete == '{0}' & NumberOfLetters == {1}"
for concreteness in ("Concrete", "Abstract"):
    for nlet in epochs.metadata["NumberOfLetters"].unique():
        subset = epochs[query.format(concreteness, nlet)]
        evokeds["/".join((concreteness, str(nlet)))] = subset.average()

style_plot["linestyles"] = {"Concrete": "-", "Abstract": ":"}

fig, ax = plt.subplots(figsize=(6, 4))
mne.viz.plot_compare_evokeds(evokeds, axes=ax, **style_plot)
plt.show()


###############################################################################
# .. note::
#
#    Creating an :class:`mne.Epochs` object with metadata is done by passing
#    a :class:`pandas.DataFrame` to the ``metadata`` kwarg as follows:

data = epochs.get_data()
metadata = epochs.metadata.copy()
epochs_new = mne.EpochsArray(data, epochs.info, metadata=metadata)
