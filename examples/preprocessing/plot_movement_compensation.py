"""
==============================================
Maxwell filter data with movement compensation
==============================================

Demonstrate movement compensation on simulated data. The simulated data
contains bilateral activation of auditory cortices, repeated over 14
different head rotations (head center held fixed). See the following for
details:

    https://github.com/mne-tools/mne-misc-data/blob/master/movement/simulate.py

"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from os import path as op

import mne
from mne.preprocessing import maxwell_filter

print(__doc__)

data_path = op.join(mne.datasets.misc.data_path(verbose=True), 'movement')

pos = mne.chpi.read_head_pos(op.join(data_path, 'simulated_quats.pos'))
raw = mne.io.read_raw_fif(op.join(data_path, 'simulated_movement_raw.fif'))
raw_stat = mne.io.read_raw_fif(op.join(data_path,
                                       'simulated_stationary_raw.fif'))

##############################################################################
# Process our simulated raw data (taking into account head movements)

# extract our resulting events
events = mne.find_events(raw, stim_channel='STI 014')
events[:, 2] = 1
raw.plot(events=events)

topo_kwargs = dict(times=[0, 0.1, 0.2], ch_type='mag', vmin=-500, vmax=500)

# 0. Take average of stationary data (bilateral auditory patterns)
evoked_stat = mne.Epochs(raw_stat, events, 1, -0.2, 0.8).average()
evoked_stat.plot_topomap(title='Stationary', **topo_kwargs)

# 1. Take a naive average (smears activity)
evoked = mne.Epochs(raw, events, 1, -0.2, 0.8).average()
evoked.plot_topomap(title='Moving: naive average', **topo_kwargs)

# 2. Use raw movement compensation (restores pattern)
raw_sss = maxwell_filter(raw, head_pos=pos)
evoked_raw_mc = mne.Epochs(raw_sss, events, 1, -0.2, 0.8).average()
evoked_raw_mc.plot_topomap(title='Moving: movement compensated', **topo_kwargs)
