"""
======================================
Export Epochs to a dataframe in Pandas
======================================

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

from pandas.stats.api import rolling_mean

# turn on interactive mode
pl.ion()

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')
raw.info['bads'] = ['MEG 2443', 'EEG 053']
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=True, eog=True, stim=False,
                            exclude=raw.info['bads'])

tmin, tmax, event_id = -0.2, 0.5, 1
baseline = (None, 0)
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False, reject=reject)

epochs_df = epochs.as_data_frame()

meg_chs = [c for c in epochs.ch_names if c.startswith("MEG")]

#display some channels
epochs_df.ix['epoch 11', :4].head(20)

# plot single epoch
epochs_df.ix['epoch 11'].plot(legend=False, xlim=(0, 105), title='epoch 11')

# split accroding to timeslices
grouped_tsl = epochs_df[meg_chs].groupby(level='tsl')

# then apply mean to each group and create a plot
grouped_tsl.mean().plot(legend=0, title='MEG average', xlim=(0, 105))

# apply custom numpy function
grouped_tsl.agg(np.std).plot(legend=0, title='MEG std', xlim=(0, 105))

# average then smooth using a rolling mean and finally plot.
grouped_tsl.apply(lambda x: rolling_mean(x.mean(), 10)).plot(legend=False,
                                                             xlim=(0, 105))

# apply individual functions to channels
grouped_tsl.agg({"MEG 0113": np.mean, "MEG 0213": np.median})

# investigate epochs
grouped_epochs = epochs_df[meg_chs].groupby(level='epochs')

subgroup = grouped_epochs.max().ix['Epoch 1':'Epoch 12']

subgroup.plot(kind='barh', legend=0)

# create string table for dumping into file.
result_table = (subgroup * 1e15).to_string()

# investigate a specific channel's std across epochs
chn = "MEG 0113"
grouped_epochs.std()[chn].plot(title=chn, xlim=(0, 55))
