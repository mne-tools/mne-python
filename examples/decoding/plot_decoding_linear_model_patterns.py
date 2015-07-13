"""
====================================================================
Decoding in sensor space data using the linear models 
====================================================================

Decoding applied to MEG data in sensor space using a linear classifier
Here the model coefiscient are interpreted using [1].


[1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., 
Blankertz, B., & Bießmann, F. (2014). On the interpretation of 
weight vectors of linear models in multivariate neuroimaging. 
NeuroImage, 87, 96–110. doi:10.1016/j.neuroimage.2013.10.067
"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne import io
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1]

import sklearn.linear_model as lm
import sklearn.svm as svm
from mne.decoding.classifier import compute_patterns

# computes patterns estimated with a ridge classifier
ridge  = lm.RidgeClassifier()
ridge_patterns = compute_patterns(epochs, ridge)
ridge_patterns.plot_topomap()

# computes patterns estimated with a linear SVM classifier
linsvc = svm.LinearSVC()
linsvc_patterns = compute_patterns(epochs, linsvc)
linsvc_patterns.plot_topomap()

# computes patterns estimated with a logistic regression classifier
logreg = lm.LogisticRegression()
logreg_patterns = compute_patterns(epochs, logreg)
logreg_patterns.plot_topomap()

