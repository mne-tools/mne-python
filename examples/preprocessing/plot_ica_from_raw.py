"""
==================================
Compute ICA components on raw data
==================================

ICA is fit to MEG raw data.
The sources matching the ECG are automatically found and displayed.
Subsequently, artifact detection and rejection quality are assessed.
"""
print(__doc__)

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs
from mne.datasets import sample

###############################################################################
# Setup paths and prepare raw data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 45, n_jobs=2)

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.

# We pass a float value between 0 and 1 to select n_components based on the
# percentage of variance explained by the PCA components.

ica = ICA(n_components=0.90, max_pca_components=None)

###############################################################################
# 1) Fit ICA model and identify bad sources

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))

# create EOG epochs to improve detection by correlation
picks = mne.pick_types(raw.info, meg=True, eog=True)
ecg_epochs = create_ecg_epochs(raw, picks=picks)

ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)  # inds sorted!

ica.plot_scores(scores, exclude=ecg_inds)  # inspect metrics used

show_picks = np.abs(scores).argsort()[::-1][:5]  # indices of top five scores

# detected artifacts drawn in red (via exclude)
ica.plot_sources(raw, show_picks, exclude=ecg_inds, start=0., stop=3.0)
ica.plot_components(ecg_inds, colorbar=False)  # show component sensitivites

ica.exclude += ecg_inds  # mark first for exclusion

###############################################################################
# 3) check detection and visualize artifact rejection

# estimate average artifact
ecg_evoked = ecg_epochs.average()
ica.plot_sources(ecg_evoked)  # latent ECG sources + selction
ica.plot_overlay(ecg_evoked)  # overlay raw and clean ECG artifacts

# check the amplitudes do not change
ica.plot_overlay(raw)  # ECG artifacts remain

###############################################################################
# To save an ICA solution you can say:
# >>> ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
#
# Apply the solution to Raw, Epochs or Evoked like this:
# >>> ica.apply(epochs, copy=False)
