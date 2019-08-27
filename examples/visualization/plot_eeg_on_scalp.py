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
from mne.viz import plot_alignment, set_3d_view

print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans_path = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
fig = plot_alignment(raw.info, trans_path, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)
set_3d_view(figure=fig, azimuth=135, elevation=80)
