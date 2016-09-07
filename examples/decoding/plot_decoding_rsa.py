""" Representational Similarity Analysis
======================================

Representational Similarity Analysis is used to perform summary statistics
on supervised classifications where the number of classes is relatively high.
It consists in characterizing the structure of the confusion matrix to infer
the similarity between brain responses and serve as a proxy for characterizing
the space of mental representations [1-3].

In this example, we perform RSA on spiking data from [4] where the macaque was
presented to seven classes of pictures located in one of three spatial
positions.

References
----------
[1] Shepard, R. "Multidimensional scaling, tree-fitting, and clustering."
 Science 210.4468 (1980): 390-398.
[2] Laakso, A. & Cottrell, G.. "Content and cluster analysis:
 assessing representational similarity in neural systems." Philosophical
 psychology 13.1 (2000): 47-76.
[3] Kriegeskorte, N., Marieke, M., & Bandettini.  P. "Representational
 similarity analysis-connecting the branches of systems neuroscience."
 Frontiers in systems neuroscience 2 (2008): 4.
[4] Cichy, R. M., Pantazis, D., & Oliva, A. "Resolving human object recognition
in space and time. Nature neuroscience" (2014): 17(3), 455-462
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import os.path as op
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

from mne.decoding import SearchLight
from mne.io import Raw
from mne import find_events, Epochs

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.manifold import MDS

data_path = '/media/DATA/Pro/Projects/NewYork/cichy/'
###############################################################################

# Define stimulus - trigger mapping
fname = op.join(data_path, 'visual_stimuli.csv')
conds = read_csv(fname)
triggers = set(conds['trigger'])
inv_conds = dict()
for val in triggers:
    sub = conds.query('trigger == %i' % val)
    inv_conds[val] = sub['condition'].iloc[0]

# Read MEG data
fname = op.join(data_path, 'sample_subject_3_tsss_mc.fif')
raw = Raw(fname, preload=True)
events = find_events(raw)
raw.pick_types(meg='mag')
raw.filter(None, 30.)

# Epoch data
is_stim = [val in triggers for val in events[:, 2]]
events = events[np.where(is_stim)[0]]
epochs = Epochs(raw, events=events, decim=4, baseline=None)

# Extract data in numpy format
times = epochs.times
X = epochs.get_data()
le = LabelEncoder()
y = le.fit_transform([inv_conds[val] for val in epochs.events[:, 2]])

##############################################################################


def roc_auc_scores(y_true, y_pred):
    """While sklearn implements a multiclass ROC (sklearn #3298)"""
    from sklearn.metrics import roc_auc_score
    auc = 0.
    classes = set(y_true)
    for ii, this_class in enumerate(classes):
        auc += roc_auc_score(y_true == this_class, y_pred[:, ii])
    return auc / len(classes)

# Decode over time to identify when information peaks
# Classifier
clf = make_pipeline(StandardScaler(), LogisticRegression())

# Search light across time
sl = SearchLight(clf, n_jobs=-1)

# Cross validation
n_folds = 5
cv = StratifiedKFold(n_folds)

n_classes = len(le.classes_)
n_epochs, n_channels, n_times = X.shape
scores = np.zeros((n_folds, n_times))

scores = np.zeros((n_folds, n_times))
for fold, (train, test) in enumerate(cv.split(X, y)):
    print(fold)
    sl.fit(X[train], y[train])
    y_pred = sl.predict_proba(X[test])
    for time_sample, y_pred_t in enumerate(y_pred.transpose(1, 0, 2)):
        scores[fold, time_sample] = roc_auc_scores(y[test], y_pred_t)

fig, ax = plt.subplots(1)
ax.plot(times, scores.mean(0))
ax.axhline(.5, color='k', linestyle=':', label='chance')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('<AUC>')
plt.show()

##############################################################################
# Representational Similarity Analysis is a neuroimaging-specific
# appelation to refer to statistics applied to the confusion matrix.
y_pred = np.zeros((n_epochs, n_classes))
cv = StratifiedShuffleSplit(200)
toi = 32
cm = np.zeros((n_classes, n_classes))
toi = np.argmax(np.sum(epochs.average().data ** 2, axis=0))
for train, test in cv.split(X, y):
    clf.fit(X[train, :, toi], y[train])
    y_pred = clf.predict(X[test, :, toi])
    cm += confusion_matrix(y[test], y_pred, labels=range(n_classes))
cm /= sum(cm)

fig, ax = plt.subplots(1)
im = ax.matshow(cm, cmap='plasma')
ax.set_yticks(range(n_classes))
ax.set_yticklabels(le.classes_)
ax.set_xticks(range(n_classes))
ax.set_xticklabels(le.classes_, rotation=40, ha='left')

plt.colorbar(im)
plt.show()


##############################################################################
# In RSA, it is common to summarize the confusion matrix with a
# dimensionality reduction. It is common to use multi-dimensional
# scaling for this, as it was historically introduced by Shepard
# who proposed to measure representation via the confusion matrix.
mds = MDS(2)
summary = mds.fit_transform(cm)
cmap = plt.get_cmap('rainbow')
colors = dict(zip(le.classes_, cmap(np.linspace(0., 1., len(le.classes_)))))
labels = list()
for condition, (x, y) in zip(le.classes_, summary):
    labels.append(condition)
    plt.scatter(x, y, s=100, facecolors=colors[condition],
                label=condition, edgecolors='k')
plt.axis('off')
plt.legend()
