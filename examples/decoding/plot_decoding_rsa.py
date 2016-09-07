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

from mne import find_events, Epochs, pick_types
from mne.io import Raw

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
fname = op.join(data_path, 'sample_subject_0_tsss_mc.fif')
raw = Raw(fname)
picks = pick_types(raw.info, meg='mag')[::2]
events = find_events(raw, min_duration=.002)

# Select & Format event values to match ad-hoc categories
events = np.array([ev for ev in events if ev[2] in set(conds['trigger'])])
# le = LabelEncoder()
# events[:, 2] = le.fit_transform([inv_conds[val] for val in events[:, 2]])

# Epoch data
is_stim = [val in triggers for val in events[:, 2]]
events = events[np.where(is_stim)[0]]
epochs = Epochs(raw, events=events, decim=4, baseline=None, picks=picks,
                tmin=.050, tmax=.800)

##############################################################################
# Representational Similarity Analysis is a neuroimaging-specific
# appelation to refer to statistics applied to the confusion matrix.

# Classify on the average ERF
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(solver='lbfgs'))
X = epochs.get_data().mean(2)
y = epochs.events[:, 2]

n_folds = 5
classes = set(y)
cv = StratifiedKFold(n_folds, random_state=0, shuffle=True)

# Compute confusion matrix for each cross-validation fold
y_pred = np.zeros((len(y), len(classes)))
for train, test in cv.split(X, y):
    # Fit
    clf.fit(X[train], y[train])
    # Predict
    y_pred[test] = clf.predict_proba(X[test])

# Compute confusion matrix using AUC
cm = np.zeros((len(classes), len(classes)))
for ii, train_class in enumerate(classes):
    for jj in range(ii, len(classes)):
        cm[ii, jj] = roc_auc_score(y == train_class, y_pred[:, jj])
        cm[jj, ii] = cm[ii, jj]
cm /= n_folds

# Plot with formatted class names for easier reading
names = [inv_conds[val] for val in classes]
sparse_names = [name if next_name != name else ''
                for name, next_name in zip(names[1:], names)]
fig, ax = plt.subplots(1)
im = ax.matshow(cm, cmap='plasma')
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(sparse_names)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(sparse_names, rotation=40, ha='left')
for ii, name in enumerate(sparse_names):
    if name != '':
        ax.axhline(ii, color='k')
        ax.axvline(ii, color='k')
plt.colorbar(im)
plt.show()

##############################################################################
# Confusion matrix related to mental representations have been historically
# summarized with dimensionality reduction using multi-dimensional scaling [1].
# See how the face samples cluster together
fig, ax = plt.subplots(1)
mds = MDS(2, random_state=0)
summary = mds.fit_transform(cm)
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0., 1., len(set(names))))
for color, name in zip(colors, set(names)):
    sel = np.where([this_name == name for this_name in names])[0]
    if name == 'human face':
        ax.plot(summary[sel, 0], summary[sel, 1], color='k', zorder=0)
    ax.scatter(summary[sel, 0], summary[sel, 1],
               s=100, facecolors=color, label=name, edgecolors='k')
ax.axis('off')
ax.legend(loc='best', scatterpoints=1, ncol=2)
plt.tight_layout()
plt.show()
