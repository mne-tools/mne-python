"""
=======================
Maxwell filter raw data
=======================

This example shows how to process M/EEG data with Maxwell filtering
in mne-python.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.preprocessing import maxwell_filter

print(__doc__)

data_path = mne.datasets.sample.data_path()

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
ctc_fname = data_path + '/SSS/ct_sparse_mgh.fif'
fine_cal_fname = data_path + '/SSS/sss_cal_mgh.dat'

# Preprocess with Maxwell filtering
raw = mne.io.Raw(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053', 'MEG 1032', 'MEG 2313']  # set bads
# Here we don't use tSSS (set st_duration) because MGH data is very clean
raw_sss = maxwell_filter(raw, cross_talk=ctc_fname, calibration=fine_cal_fname)

# Select events to extract epochs from, pick M/EEG channels, and plot evoked
tmin, tmax = -0.2, 0.5
event_id = {'Auditory/Left': 1}
events = mne.find_events(raw, 'STI 014')
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       include=[], exclude='bads')
for r, kind in zip((raw, raw_sss), ('Raw data', 'Maxwell filtered data')):
    epochs = mne.Epochs(r, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(eog=150e-6),
                        preload=False)
    evoked = epochs.average()
    evoked.plot(window_title=kind)
