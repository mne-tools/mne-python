"""
===========================================================================
Motor imagery decoding from EEG data using Riemannian Geometry
===========================================================================

Decoding of motor imagery applied to EEG data with Riemannian geometry.
For each trials, covariance matrices are estimated, and classified in the
Riemannian manifold using 2 different methods :
    * MDM : Nereast centroid using the Riemannian metric [4]
    * TS + LR : Tangent space projection to map the Riemannian manifold
    to its (euclidean) tangent space, followed by a logistic regression.

Both methods are compared to the classifier applied on features extracted
on CSP [1] filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]

The EEGBCI dataset is documented in [2]
The data set is available at PhysioNet [3]

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

[2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
    Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
    (BCI) System. IEEE TBME 51(6):1034-1043

[3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
    Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
    PhysioToolkit, and PhysioNet: Components of a New Research Resource for
    Complex Physiologic Signals. Circulation 101(23):e215-e220

[4] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry,
    in IEEE Transactions on Biomedical Engineering,
    vol. 59, no. 4, p. 920-928, 2012.

"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

# generic import
import numpy as np
from pylab import plt

# mne import
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP

# pyriemann import
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import covariances

# sklearn imports
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

###############################################################################
# Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1., 2.
event_id = dict(hands=2, feet=3)
subject = 7
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_files = [read_raw_edf(f, preload=True)
             for f in eegbci.load_data(subject, runs)]
raw = concatenate_raws(raw_files)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
# Subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7., 30., method='iir', picks=picks)

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True, add_eeg_ref=False, verbose=False)
labels = epochs.events[:, -1] - 2

# cross validation
cv = KFold(len(labels), 10, shuffle=True, random_state=42)
# get epochs
epochs_data_train = epochs.get_data()

# compute covariance matrices
cov_data_train = covariances(epochs_data_train)

###############################################################################
# Classification with Minimum distance to mean
mdm = MDM()

# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("MDM Classification accuracy:       %f / Chance level: %f"
      % (np.mean(scores), class_balance))

###############################################################################
# Classification with Tangent Space Logistic Regression
ts = TangentSpace(metric='riemann')
lr = LogisticRegression(penalty='l2')

clf = Pipeline([('TS', ts), ('LR', lr)])
# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("TS + LR Classification accuracy:   %f / Chance level: %f"
      % (np.mean(scores), class_balance))

###############################################################################
# Classification with CSP + linear discrimant analysis

# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg='lws', log=True)

clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("CSP + LDA Classification accuracy: %f / Chance level: %f"
      % (np.mean(scores), class_balance))

# plot prototyped covariance matrices
mdm = MDM()
mdm.fit(cov_data_train, labels)
fig, axe = plt.subplots(1, 2)
ch_labels = [raw.ch_names[i].strip('.') for i in picks]

axe[0].matshow(mdm.covmeans[0])
plt.setp(axe[0], yticks=range(len(ch_labels)), yticklabels=ch_labels,
         xticks=[0], xticklabels=[''])
axe[0].set_title('Hand mean covariance matrix')

axe[1].matshow(mdm.covmeans[1])
plt.setp(axe[1], yticks=range(len(ch_labels)), yticklabels=ch_labels,
         xticks=[0], xticklabels=[''])
axe[1].set_title('Feet mean covariance matrix')
plt.show()
