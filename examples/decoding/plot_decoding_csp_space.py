"""
==========================
Decoding in sensor space data using the Common Spatial Pattern (CSP).
algorithm [1] (see http://en.wikipedia.org/wiki/Common_spatial_pattern)
==========================

Decoding applied to MEG data in sensor space decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

print __doc__
import pylab as pl
import numpy as np

import mne
from mne import fiff
from mne.datasets import sample

data_path = sample.data_path()

pl.close('all')

###############################################################################
# Set parameters
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

# Setup for reading the raw data
raw = fiff.Raw(raw_fname, preload=True)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:,-1]
evoked = epochs.average()
###############################################################################
# Decoding in sensor space using a linear SVM

from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from mne.csp import CSP

clf = SVC(C=1, kernel='linear')
csp = CSP()
# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2)
# take the first and last components (aka largest eigen values)
csp_components = np.array([0, 1, -2, -1])
csp_scores = []

for train_idx, test_idx in cv:
    
    train_epochs = epochs[train_idx]
    train_labels = labels[train_idx]
    
    test_epochs = epochs[test_idx]
    test_labels = labels[test_idx]
    # compute spatial filters
    csp.decompose_epochs([train_epochs[train_labels==1],
                          train_epochs[train_labels==3]])
    
    # compute sources for train and test data
    train_data = csp.get_sources_epochs(train_epochs,csp_components)
    test_data  = csp.get_sources_epochs(test_epochs,csp_components)
    # compute features (mean band power)
    train_data = (train_data**2).mean(-1)
    test_data  = (test_data**2).mean(-1)
    
    # Standardize features
    m_train = train_data.mean(axis=0)
    s_train = train_data.std(axis=0)
    train_data = (train_data - m_train)/s_train
    test_data = (test_data - m_train)/s_train
    
    # fit classifier
    clf.fit(train_data, train_labels)
    y_true, y_pred = test_labels, clf.predict(test_data)
    
    csp_scores.append(100*(y_true == y_pred).mean())
    # plot csp patterns
    evoked.data = csp.patterns.T
    evoked.times = np.arange(evoked.data.shape[0])
    evoked.plot_topomap(times = [0,1,201,202],ch_type='grad',colorbar=False)
    
## TODO: plot csp patterns!