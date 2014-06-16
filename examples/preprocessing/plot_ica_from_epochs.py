"""
================================
Compute ICA components on epochs
================================

ICA is fit to MEG raw data.
The sources matching the EOG are automatically found and displayed.
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
from mne.preprocessing import ICA, create_eog_epochs
from mne.datasets import sample

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 30, method='iir')
picks = mne.pick_types(raw.info, meg=True, eeg=False, ecg=False,
                       eog=True, stim=False, exclude='bads')

tmin, tmax, event_id = -0.2, 0.5, 1
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=(None, 0), preload=True, reject=None)

###############################################################################
# 1) Fit ICA model

ica = ICA(n_components=0.99).fit(epochs)

###############################################################################
# 2) Find eog Artifacts

# generate eog epochs to improve detection by correlation
eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks)


eog_inds, scores = ica.find_bads_eog(eog_epochs)
ica.plot_scores(scores, exclude=eog_inds)

title = 'Sources related to %s artifacts (red)'
show_picks = np.abs(scores).argsort()[::-1][:5]

ica.plot_sources(epochs, show_picks, exclude=eog_inds, title=title % 'eog')
ica.plot_components(eog_inds, title=title % 'eog')

ica.exclude += eog_inds[:1]  # mark bad components, rely on first

###############################################################################
# 3) Assess component selection and unmixing quality

# estimate average artifact
eog_evoked = eog_epochs.average()
ica.plot_sources(eog_evoked)  # plot EOG sources + selection
ica.plot_overlay(eog_evoked)  # plot EOG cleaning

# check effect on ERF of interest
ica.plot_overlay(epochs.average())  # plot remaining ERF
