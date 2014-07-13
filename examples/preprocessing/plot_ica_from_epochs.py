"""
================================
Compute ICA components on epochs
================================

ICA is fit to MEG raw data.
We assume that the non-stationary EOG artifacts have already been removed.
The sources matching the ECG are automatically found and displayed.
Subsequently, artefact detection and rejection quality are assessed.
Finally, the impact on the evoked ERF is visualized.
"""
print(__doc__)

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne.io import Raw
from mne.preprocessing import ICA, create_ecg_epochs
from mne.datasets import sample

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 30, method='iir')
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, ecg=True,
                       stim=False, exclude='bads')

# longer + more epochs for more artifact exposure
events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
reject = dict(eog=250e-6)
tmin, tmax = -0.5, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=(None, 0), preload=True, reject=reject)

###############################################################################
# 1) Fit ICA model using the FastICA algorithm

ica = ICA(n_components=0.95, method='fastica').fit(epochs)

###############################################################################
# 2) Find ECG Artifacts

# generate ECG epochs to improve detection by correlation
ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)

ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)
ica.plot_scores(scores, exclude=ecg_inds)

title = 'Sources related to %s artifacts (red)'
show_picks = np.abs(scores).argsort()[::-1][:5]

ica.plot_sources(epochs, show_picks, exclude=ecg_inds, title=title % 'ecg')
ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True)

ica.exclude += ecg_inds[:3]  # by default we expect 3 reliable ECG components

###############################################################################
# 3) Assess component selection and unmixing quality

# estimate average artifact
ecg_evoked = ecg_epochs.average()
ica.plot_sources(ecg_evoked)  # plot ECG sources + selection
ica.plot_overlay(ecg_evoked)  # plot ECG cleaning

# check effect on ERF of interest
epochs.crop(-.2, None)  # crop to baseline of interest
ica.plot_overlay(epochs['aud_l'].average())  # plot remaining left auditory ERF
