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
epochs_df.ix[:, :10].head(20)

# split timeslices
grouped_tsl = epochs_df[meg_chs].groupby(level='tsl')

# then create a quick average plot
grouped_tsl.mean().plot(legend=0)

# or a trellis plot on a few channels
grouped_tsl.mean()[meg_chs[:10]].plot(subplots=1)

# use median instead
grouped_tsl.median().plot(legend=0)

# use custom numpy function
grouped_tsl.agg(np.std).plot(legend=0)

# average then smooth using a rolling mean and finally plot in one single line!
grouped_tsl.apply(lambda x: rolling_mean(x.mean(), 10)).plot(legend=0)

# apply different functio for channels
grouped_tsl.agg({"MEG 0113": np.mean, "MEG 0213": np.median})

# investigate epochs and create string table for dumping into file
grouped_epochs = epochs_df[meg_chs].groupby(level='epochs')

result_table = (grouped_epochs.max().ix[:, 1:3] * 1e15).to_string()

# investigate a specific channel's std across epochs
grouped_epochs.std()["MEG 0113"].plot()

grouped_tsl.agg(np.std).plot(legend=0)

pl.show()
