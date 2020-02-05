"""
.. _tut-epochs-dataframe:

Exporting Epochs to Pandas DataFrames
=====================================

This tutorial shows how to export the data in :class:`~mne.Epochs` objects to a
:class:`Pandas DataFrame <pandas.DataFrame>`, and applies a typical Pandas
:doc:`split-apply-combine <pandas:user_guide/groupby>` workflow to examine the
latencies of the response maxima across epochs and conditions.

.. contents:: Page contents
   :local:
   :depth: 2

We'll use the :ref:`sample-dataset`, but load a version of the raw file that
has already been filtered and downsampled, and has an average reference applied
to its EEG channels. As usual we'll start by importing the modules we need and
loading the data:
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

###############################################################################
# Next we'll load a list of events from file, map them to condition names with
# an event dictionary, set some signal rejection thresholds (cf.
# :ref:`tut-reject-epochs-section`), and segment the continuous data into
# epochs:

sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_filt-0-40_raw-eve.fif')
events = mne.read_events(sample_data_events_file)

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}

reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 μV
                       eog=200e-6)       # 200 μV

tmin, tmax = (-0.2, 0.5)  # epoch from 200 ms before event to 500 ms after it
baseline = (None, 0)      # baseline period from start of epoch to time=0

epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True,
                    baseline=baseline, reject=reject_criteria, preload=True)

###############################################################################
# Converting an ``Epochs`` object to a ``DataFrame``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Once we have our :class:`~mne.Epochs` object, converting it to a
# :class:`~pandas.DataFrame` is simple: just call :meth:`epochs.to_data_frame()
# <mne.Epochs.to_data_frame>`. Each channel's data will be a column of the new
# :class:`~pandas.DataFrame`, alongside three additional columns of event name,
# epoch number, and sample time (converted from seconds to milliseconds and
# then rounded to an integer):

df = epochs.to_data_frame()
df.head()

###############################################################################
# Note that, by default, channel measurement values are scaled so that EEG data
# are converted to μV, magnetometer data are converted to fT, and gradiometer
# data are converted to fT/cm. These scalings can be customized through the
# ``scalings`` parameter, or suppressed by passing ``scalings=dict(eeg=1,
# mag=1, grad=1)``.
#
# If you don't want time converted to milliseconds, you can pass
# ``time_format=None`` to keep time as a :class:`float` value in seconds, or
# convert it to a :class:`~pandas.Timedelta` value via
# ``time_format='timedelta'``.

df = epochs.to_data_frame(time_format=None,
                          scalings=dict(eeg=1, mag=1, grad=1))

###############################################################################
# It is also possible to move one or more of the indicator columns (event name,
# epoch number, and sample time) into the :ref:`index <pandas:indexing>`, by
# passing a string or list of strings as the ``index`` parameter.

df = epochs.to_data_frame(index=['condition', 'epoch'],
                          time_format='timedelta')
df.head()

###############################################################################
# Another parameter, ``long_format``, determines whether each channel's data is
# in a separate column of the :class:`~pandas.DataFrame`
# (``long_format=False``), or whether the measured values are pivoted into a
# single ``'value'`` column with an extra indicator column for the channel name
# (``long_format=True``).

df = epochs.to_data_frame(time_format=None, index='condition',
                          long_format=True)
df.head()

###############################################################################
# This can be helpful when passing the :class:`~pandas.DataFrame` to other
# modules for subsequent analysis or plotting. For example:

# plot a line for mean (across epochs in the chosen condition) in each channel,
# with confidence band for variability across epochs:
sns.lineplot(x='time', y='value', hue='channel', data=df.loc['auditory/left'],
             legend=False)

# TODO resume revisions here


###############################################################################
# Explore Pandas MultiIndex

# Pandas is using a MultiIndex or hierarchical index to handle higher
# dimensionality while at the same time representing data in a flat 2d manner.

print(df.index.names, df.index.levels)

# Inspecting the index object unveils that 'epoch', 'time' are used
# for subsetting data. We can take advantage of that by using the
# .loc attribute, where in this case the first position indexes the MultiIndex
# and the second the columns, that is, channels.

# Plot some channels across the first three epochs
xticks, sel = np.arange(3, 600, 120), meg_chs[:15]
df.loc[:3, sel].plot(xticks=xticks)
mne.viz.tight_layout()

# slice the time starting at t0 in epoch 2 and ending 500ms after
# the base line in epoch 3. Note that the second part of the tuple
# represents time in milliseconds from stimulus onset.
df.loc[(1, 0):(3, 500), sel].plot(xticks=xticks)
mne.viz.tight_layout()

# Note: For convenience the index was converted from floating point values
# to integer values. To restore the original values you can e.g. say
# df['times'] = np.tile(epoch.times, len(epochs_times)

# We now reset the index of the DataFrame to expose some Pandas
# pivoting functionality. To simplify the groupby operation we
# we drop the indices to treat epoch and time as categroial factors.

df = df.reset_index()

# The ensuing DataFrame then is split into subsets reflecting a crossing
# between condition and trial number. The idea is that we can broadcast
# operations into each cell simultaneously.

factors = ['condition', 'epoch']
sel = factors + ['MEG 1332', 'MEG 1342']
grouped = df[sel].groupby(factors)

# To make the plot labels more readable let's edit the values of 'condition'.
df.condition = df.condition.apply(lambda name: name + ' ')

# Now we compare the mean of two channels response across conditions.
grouped.mean().plot(kind='bar', stacked=True, title='Mean MEG Response',
                    color=['steelblue', 'orange'])
mne.viz.tight_layout()

# We can even accomplish more complicated tasks in a few lines calling
# apply method and passing a function. Assume we wanted to know the time
# slice of the maximum response for each condition.

max_latency = grouped[sel[2]].apply(lambda x: df.time[x.idxmax()])

print(max_latency)

# Then make the plot labels more readable let's edit the values of 'condition'.
df.condition = df.condition.apply(lambda name: name + ' ')

plt.figure()
max_latency.plot(kind='barh', title='Latency of Maximum Response',
                 color=['steelblue'])
mne.viz.tight_layout()

# Finally, we will again remove the index to create a proper data table that
# can be used with statistical packages like statsmodels or R.

final_df = max_latency.reset_index()
final_df.rename(columns={0: sel[2]})  # as the index is oblivious of names.

# The index is now written into regular columns so it can be used as factor.
print(final_df)

plt.show()

# To save as csv file, uncomment the next line.
# final_df.to_csv('my_epochs.csv')

# Note. Data Frames can be easily concatenated, e.g., across subjects.
# E.g. say:
#
# import pandas as pd
# group = pd.concat([df_1, df_2])
# group['subject'] = np.r_[np.ones(len(df_1)), np.ones(len(df_2)) + 1]

##############################################################################
# Long-format dataframes

# Many statistical modelling functions expect data in a long format
# where each row is one observation at a unique coordinate of factors
# such as sensors, conditions, subjects etc.
df_long = epochs.to_data_frame(long_format=True)
print(df_long.head())

# Here the MEG or EEG signal appears in the column "observation".
# The total length is therefore the number of channels times the time points.
print(len(df_long), "=", epochs.get_data().size)

# To simplify subsetting and filtering a channwel type column is added.
print(df_long.query("ch_type == 'eeg'").head())

# Note that some of the columns are transformed to "category" data types.
print(df_long.dtypes)

##############################################################################
# The pandas exporter facilitates processing MNE outputs in R:
# https://mne-tools.github.io/mne-r/articles/plot_evoked_multilevel_model.html
