"""
================================
Compute ICA components on epochs
================================

ICA is fit to MEG raw data.
We assume that the non-stationary EOG artifacts have already been removed.
The sources matching the ECG are automatically found and displayed.

.. note:: This example does quite a bit of processing, so even on a
          fast machine it can take about a minute to complete.
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.preprocessing import ICA, create_ecg_epochs
from mne.datasets import sample

print(__doc__)

###############################################################################
# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter
# - epoching -0.2 to 0.5 seconds with respect to events
# - rejection based on peak-to-peak amplitude

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname)
raw.pick_types(meg=True, eeg=False, exclude='bads', stim=True).load_data()
raw.filter(1, 30, fir_design='firwin')

# peak-to-peak amplitude rejection parameters
reject = dict(grad=4000e-13, mag=4e-12)
# longer + more epochs for more artifact exposure
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5,
                    reject=reject)

###############################################################################
# Fit ICA model using the FastICA algorithm, detect and plot components
# explaining ECG artifacts.

ica = ICA(n_components=0.95, method='fastica').fit(epochs)

ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, threshold='auto')

ica.plot_components(ecg_inds)

###############################################################################
# Plot properties of ECG components:
ica.plot_properties(epochs, picks=ecg_inds)

###############################################################################
# Plot the estimated source of detected ECG related components
ica.plot_sources(raw, picks=ecg_inds)
