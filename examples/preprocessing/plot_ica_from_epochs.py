"""
================================
Compute ICA components on epochs
================================

ICA is fit to MEG raw data.
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
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       ecg=True, stim=False, exclude='bads')

tmin, tmax, event_id = -0.2, 0.5, 1
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=(None, 0), preload=True, reject=None)

###############################################################################
# 1) Fit ICA model

ica = ICA(n_components=0.99).fit(epochs)

###############################################################################
# 2) Find ECG Artifacts

# generate ECG epochs to improve detection by correlation
ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)


ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, ch_name='MEG 1531')
ica.plot_scores(scores, exclude=ecg_inds)

title = 'Sources related to %s artifacts (red)'
show_picks = np.abs(scores).argsort()[::-1][:5]

ica.plot_sources(epochs, show_picks, exclude=ecg_inds, title=title % 'ECG')
ica.plot_components(ecg_inds, title=title % 'ECG')

ica.exclude += ecg_inds[:2]  # mark bad components, rely on first two

###############################################################################
# 3) Assess component selection and unmixing quality

# estimate average artifact
ecg_evoked = ecg_epochs.average()
ica.plot_sources(ecg_evoked)  # plot ECG sources + selection
ica.plot_overlay(ecg_evoked)  # plot ecg cleaning

# check effect on ERF of interest
ica.plot_overlay(epochs.average())  # plot remaining ERF
