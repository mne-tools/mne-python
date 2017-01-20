"""
====================================
Representational Similarity Analysis
====================================

Representational Similarity Analysis is used to perform summary statistics
on supervised classifications where the number of classes is relatively high.
It consists in characterizing the structure of the confusion matrix to infer
the similarity between brain responses and serve as a proxy for characterizing
the space of mental representations [1-3].

In this example, we perform RSA on responses to 92 object images. Subjects were
presented images of human, animal and inanimate objects [4].

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
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import MDS

import mne
from mne.io import read_raw_fif, concatenate_raws
from mne.datasets import visual_92_categories

print(__doc__)

data_path = visual_92_categories.data_path()

# Define stimulus - trigger mapping
fname = op.join(data_path, 'visual_stimuli.csv')
conds = read_csv(fname)
print(conds.head(5))

##############################################################################
# Let's restrict the number of conditions to speed up computation
max_trigger = 24
conds = conds[:max_trigger]  # take only the first 24 rows

##############################################################################
# Define stimulus - trigger mapping
conditions = ['/'.join(map(str,
                           list(c[:2]) + [('not-' if i == 0 else '') +
                                          conds.columns[k]
                                          for k, i in enumerate(c[2:], 2)]))
              for c in conds.values]
print(conditions[:10])

##############################################################################
# Let's make the event_id dictionary
event_id = dict(zip(conditions, map(int, conds.trigger + 1)))
event_id['0/human bodypart/human/not-face/animal/natural']

##############################################################################
# Read MEG data
n_runs = 4  # 4 for full data (use less to speed up computations)
fname = op.join(data_path, 'sample_subject_%i_tsss_mc.fif')
raws = [read_raw_fif(fname % block) for block in range(n_runs)]
raw = concatenate_raws(raws)

events = mne.find_events(raw, min_duration=.002, verbose=True)
print(events[:10])

events = events[events[:, 2] <= max_trigger]
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])

##############################################################################
# Epoch data
picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(raw, events=events, event_id=event_id, baseline=None,
                    picks=picks, tmin=-.1, tmax=.500, preload=True)

##############################################################################
# Let's plot some conditions
epochs['face'].average().plot()
epochs['not-face'].average().plot()

##############################################################################
# Representational Similarity Analysis (RSA) is a neuroimaging-specific
# appelation to refer to statistics applied to the confusion matrix
# also referred to as the representational dissimilarity matrices (RDM).
#
# Compared to the approach from Cichy et al. we'll use a multiclass
# classifier (Multinomial Logistic Regression) while the paper uses
# all pairwise binary classification task to make the RDM.
# Also we use here the ROC-AUC as performance metric while the
# paper uses accuracy. Finally here for the sake of time we use
# RSA on a window of data while Cichy et al. did it for all time
# instants separately.

# Classify using the average signal in the window 50ms to 300ms
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(C=1, solver='lbfgs'))
X = epochs.copy().crop(0.05, 0.3).get_data().mean(axis=2)
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

print(y_pred[:2])


##############################################################################
# Compute confusion matrix using ROC-AUC
cm = np.zeros((len(classes), len(classes)))
for ii, train_class in enumerate(classes):
    for jj in range(ii, len(classes)):
        cm[ii, jj] = roc_auc_score(y == train_class, y_pred[:, jj])
        cm[jj, ii] = cm[ii, jj]

# Format class names for centered plotting
names = np.array([conds[conds.trigger == (val - 1)].condition.iloc[0]
                  for val in classes])
sparse_names = np.copy(names)
n = 0
for ii, name in enumerate(names):
    sparse_names[ii] = '' if n != (sum(names == name) // 2) else name
    n = 0 if ii < (len(names) - 1) and names[ii + 1] != name else n + 1

##############################################################################
# Plot
fig, ax = plt.subplots(1)
im = ax.matshow(cm, cmap='RdBu_r', clim=[0.3, 0.7])
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(sparse_names)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(sparse_names, rotation=40, ha='left')
for ii, (name, next_name) in enumerate(zip(names, names[1:])):
    if name != next_name:
        ax.axhline(ii + 0.5, color='k')
        ax.axvline(ii + 0.5, color='k')
plt.colorbar(im)
plt.tight_layout()
plt.show()

##############################################################################
# Confusion matrix related to mental representations have been historically
# summarized with dimensionality reduction using multi-dimensional scaling [1].
# See how the face samples cluster together.
fig, ax = plt.subplots(1)
mds = MDS(2, random_state=0, dissimilarity='precomputed')
chance = 0.5
summary = mds.fit_transform(chance - cm)
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0., 1., len(set(names))))
for color, name in zip(colors, set(names)):
    sel = np.where([this_name == name for this_name in names])[0]
    size = 500 if name == 'human face' else 100
    ax.scatter(summary[sel, 0], summary[sel, 1], s=size,
               facecolors=color, label=name, edgecolors='k')
ax.axis('off')
ax.legend(loc='lower right', scatterpoints=1, ncol=2)
plt.tight_layout()
plt.show()
