"""
======================================
Export Epochs to a data frame in Pandas
======================================

In this example the pandas exporter will be used to prodcue a DataFrame.
After exploring some basic features of a DataFrame a split-apply-combine
combine work flow will be conducted to examine the latencies of the response
maximums across epochs and conditions.

Short Pandas Primer
----------------------------

Pandas Data Frames
~~~~~~~~~~~~~~~~~~
A data frame can be thought of as a product between a matrix, a list and a dict:
It knows about linear algebra and element-wise operations but is size mutable
and allows for labeled access to its data. In addition, the pandas data frame
class provides many useful methods for restructuring, reshaping and visualizing
data. As most methods return data frame instances, operations can be chained
with ease; this allows to write efficient one-liners.
Taken together, these features qualify data frames for inter operation with
databases and for interactive data exploration / analysis.
Additionally, pandas interfaces with the R statistical computing language that
covers a huge amount of statistical functionality.

Export Options
~~~~~~~~~~~~~~
The pandas exporter comes with a few options worth being commented.

Pandas DataFrame objects use so a called hierarchical index. This can be
thought of as array of unique tuples, in our case, representing the higher
dimensional MEG data in a 2D data table. The column names are the
channel names from the epoch object. The channels can be accessed like entries
of a dictionary:

    df['MEG 2333']

Epochs and time slices can be accessed with the .ix method:

    epochs_df.ix[(1, 2), 'MEG 2333']

However, it is also possible to include this index as regular categorial data
columns, which yields a long table format typically used for repeated measure
designs. To take control of this feature, on export, you can specify which
of the three dimensions 'condition', 'epoch' and 'time' is passed to the Pandas
index using the index parameter. Note that this decision is revertible any time,
as demonstrated below.

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
    can be used to manipulate the underlying data separately but simultaneously.
    Finally reset_index can be used to combine the results back into a data
    frame.
plot : wrapper around plt.plot
    However it comes with some special options. For examples see below.
shape : shape attribute
    gets the dimensions of the data frame.

Reference
~~~~~~~~~
More information and additional introductory materials can be found at the pandas
doc sites: http://pandas.pydata.org/pandas-docs/stable/
"""
# Author: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import mne
import pylab as pl
import numpy as np
from mne.fiff import Raw
from mne.datasets import sample

# turn on interactive mode
pl.ion()

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

raw = Raw(raw_fname)

# For simplicity we will only consider the first 10 epochs
events = mne.read_events(event_fname)[:10]

raw.info['bads'] = ['MEG 2443']
picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=False, exclude=raw.info['bads'])

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
# this is optional. By passing None you could also print out all nesting factors
# in a long table style commonly used for analyzing repeated measure designs.

index, scale_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)

df = epochs.as_data_frame(picks=None, scalings=scalings, scale_time=scale_time,
                          index=index)

# Create MEG channel selector and drop EOG channel.
meg_chs = [c for c in df.columns if 'MEG' in c]

df.pop('EOG 061')  # this works just like with a list.

###############################################################################
# Explore Pandas MultiIndex

# Pandas is using a MultiIndex or hierarchical index to handle higher
# dimensionality while at the same time representing data in a flat 2d manner.

print df.index.names, df.index.levels

# Inspecting the index object unveils that 'epoch', 'time' are used
# for subsetting data. We can take advantage of that by using the
# .ix attribute, where in this case the first position indexes the MultiIndex
# and the second the columns, that is, channels.

# plot some channels across for the first three epochs using

xticks, sel = np.arange(3, 600, 120), meg_chs[:15]
df.ix[:3, sel].plot(xticks=xticks)

# slice the time starting at t0 in epoch 2 and ending 500ms after
# the base line in epoch 3.
pl.figure()
df.ix[(1, 0):(3, 500), sel].plot(xticks=xticks)

# Note: To take more advantage of the index was set from floating values
# to int values. To restore the original values you can e.g. say
# df['times'] = np.tile(epoch.times, len(epochs_times)

# We now want to add 'condition' to the DataFrame to make expose
# Pandas pivoting functionality.

df.set_index('condition', append=True, inplace=True)

# The DataFrame now is split into subsets reflecting a
# crossing between condition and trial number.
# For demonstration purposes we only take the first 10 epochs
# The idea is that we can broadcast operations into each cell simultaneously.

grouped = df.groupby(level=['condition', 'epoch'])

# you can think of it as a dict:
print grouped.groups['visual_r', 6][:10]

# print condition aggregate statistics for one channels
print  grouped['MEG 1332'].describe()

# plot the mean response according to condition.
pl.figure()
grouped['MEG 1332'].mean().plot(kind='bar', title='Mean MEG Response')

# We can even accomplish more complicated task with in a few lines.
# Assume we wanted to know the time slice with the maximum response

max_latency = grouped['MEG 1332'].apply(lambda x: x.index[x.argmax()][1])

print max_latency

# plot
pl.figure()
max_latency.plot(kind='barh', title='Latency of Maximum Reponse')

# finally we will remove the index to create a data table usable for
# use with statistical packages like statsmodels or R.

final_df = max_latency.reset_index()

# The index is now write into regular columns so it can be used as factor.
print final_df

# to save as csv file uncomment the next line.
# final_df.to_csv('my_epochs.csv')
