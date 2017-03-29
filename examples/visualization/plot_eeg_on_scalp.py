"""
=================================
Plotting EEG sensors on the scalp
=================================

In this example, digitized EEG sensor locations are shown on the scalp.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.viz import plot_trans
from mayavi import mlab

print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
fig = plot_trans(raw.info, trans, subject='sample', dig=False,
                 eeg_sensors=['original', 'projected'],
                 meg_sensors=[], coord_frame='head', subjects_dir=subjects_dir)
mlab.view(135, 80)
