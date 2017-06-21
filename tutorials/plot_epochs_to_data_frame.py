"""
.. _tut_io_export_pandas:

=================================
Export epochs to Pandas DataFrame
=================================

In this example the pandas exporter will be used to produce a DataFrame
object. After exploring some basic features a split-apply-combine
work flow will be conducted to examine the latencies of the response
maxima across epochs and conditions.
Note. Equivalent methods are available for raw and evoked data objects.

Short Pandas Primer
-------------------

Pandas Data Frames
~~~~~~~~~~~~~~~~~~
A data frame can be thought of as a combination of matrix, list and dict:
It knows about linear algebra and element-wise operations but is size mutable
and allows for labeled access to its data. In addition, the pandas data frame
class provides many useful methods for restructuring, reshaping and visualizing
data. As most methods return data frame instances, operations can be chained
with ease; this allows to write efficient one-liners. Technically a DataFrame
can be seen as a high-level container for numpy arrays and hence switching
back and forth between numpy arrays and DataFrames is very easy.
Taken together, these features qualify data frames for inter operation with
databases and for interactive data exploration / analysis.
Additionally, pandas interfaces with the R statistical computing language that
covers a huge amount of statistical functionality.

Export Options
~~~~~~~~~~~~~~
The pandas exporter comes with a few options worth being commented.

Pandas DataFrame objects use a so called hierarchical index. This can be
thought of as an array of unique tuples, in our case, representing the higher
dimensional MEG data in a 2D data table. The column names are the channel names
from the epoch object. The channels can be accessed like entries of a
dictionary:

    df['MEG 2333']

Epochs and time slices can be accessed with the .ix method:

    epochs_df.ix[(1, 2), 'MEG 2333']

However, it is also possible to include this index as regular categorial data
columns which yields a long table format typically used for repeated measure
designs. To take control of this feature, on export, you can specify which
of the three dimensions 'condition', 'epoch' and 'time' is passed to the Pandas
index using the index parameter. Note that this decision is revertible any
time, as demonstrated below.

Similarly, for convenience, it is possible to scale the times, e.g. from
seconds to milliseconds.

Some Instance Methods
~~~~~~~~~~~~~~~~~~~~~
Most numpy methods and many ufuncs can be found as instance methods, e.g.
mean, median, var, std, mul, , max, argmax etc.
Below an incomplete listing of additional useful data frame instance methods:

apply : apply function to data.
    Any kind of custom function can be applied to the data. In combination with
    lambda this can be very useful.
describe : quickly generate summary stats
    Very useful for exploring data.
groupby : generate subgroups and initialize a 'split-apply-combine' operation.
    Creates a group object. Subsequently, methods like apply, agg, or transform
    can be used to manipulate the underlying data separately but
    simultaneously. Finally, reset_index can be used to combine the results
    back into a data frame.
plot : wrapper around plt.plot
    However it comes with some special options. For examples see below.
shape : shape attribute
    gets the dimensions of the data frame.
values :
    return underlying numpy array.
to_records :
    export data as numpy record array.
to_dict :
    export data as dict of arrays.

Reference
~~~~~~~~~
More information and additional introductory materials can be found at the
pandas doc sites: http://pandas.pydata.org/pandas-docs/stable/
"""
# Author: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference()  # set EEG average reference

# For simplicity we will only consider the first 10 epochs
events = mne.read_events(event_fname)[:10]

# Add a bad channel
raw.info['bads'] += ['MEG 2443']
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, exclude='bads')

tmin, tmax = -0.2, 0.5
baseline = (None, 0)
reject = dict(grad=4000e-13, eog=150e-6)

event_id = dict(auditory_l=1, auditory_r=2, visual_l=3, visual_r=4)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, reject=reject)

###############################################################################
# Export DataFrame

# The following parameters will scale the channels and times plotting
# friendly. The info columns 'epoch' and 'time' will be used as hierarchical
# index whereas the condition is treated as categorial data. Note that
# this is optional. By passing None you could also print out all nesting
# factors in a long table style commonly used for analyzing repeated measure
# designs.

index, scale_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)

df = epochs.to_data_frame(picks=None, scalings=scalings, scale_time=scale_time,
                          index=index)

# Create MEG channel selector and drop EOG channel.
meg_chs = [c for c in df.columns if 'MEG' in c]

df.pop('EOG 061')  # this works just like with a list.

###############################################################################
# Explore Pandas MultiIndex

# Pandas is using a MultiIndex or hierarchical index to handle higher
# dimensionality while at the same time representing data in a flat 2d manner.

print(df.index.names, df.index.levels)

# Inspecting the index object unveils that 'epoch', 'time' are used
# for subsetting data. We can take advantage of that by using the
# .ix attribute, where in this case the first position indexes the MultiIndex
# and the second the columns, that is, channels.

# Plot some channels across the first three epochs
xticks, sel = np.arange(3, 600, 120), meg_chs[:15]
df.ix[:3, sel].plot(xticks=xticks)
mne.viz.tight_layout()

# slice the time starting at t0 in epoch 2 and ending 500ms after
# the base line in epoch 3. Note that the second part of the tuple
# represents time in milliseconds from stimulus onset.
df.ix[(1, 0):(3, 500), sel].plot(xticks=xticks)
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

max_latency = grouped[sel[2]].apply(lambda x: df.time[x.argmax()])

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
