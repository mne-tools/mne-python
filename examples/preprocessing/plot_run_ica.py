# doc:slow-example
"""
================================
Compute ICA components on epochs
================================

ICA is fit to MEG raw data.
We assume that the non-stationary EOG artifacts have already been removed.
The sources matching the ECG are automatically found and displayed.
Subsequently, artefact detection and rejection quality are assessed.
Finally, the impact on the evoked ERF is visualized.

Note that this example does quite a bit of processing, so even on a
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
# Fit ICA model using the FastICA algorithm, detect and inspect components

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 30, method='iir')
raw.pick_types(meg=True, eeg=False, exclude='bads', stim=True)

# longer + more epochs for more artifact exposure
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5)

ica = ICA(n_components=0.95, method='fastica').fit(epochs)

ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)

ica.plot_components(ecg_inds)
