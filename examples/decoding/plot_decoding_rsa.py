"""Representational Similarity Analysis is used to perform summary statistics
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
[4] Zhang, Ying, et al. "Object decoding with attention in inferior temporal
 cortex." Proceedings of the National Academy of Sciences 108.21 (2011):
 8850-8855.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas import DataFrame
from itertools import product

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from mne.decoding import SearchLight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.manifold import MDS

data_path = './data/'
files = os.listdir(data_path)

classes = ('car', 'couch', 'face', 'flower', 'guitar', 'hand', 'kiwi')
positions = ('lower', 'middle', 'upper')

###############################################################################

# FIXME Add this spiking data in MNE samples in a fif format?
# epoched matlab data available from
# http://www.readout.info/downloads/datasets/zhang-desimone-7-object-dataset/

# Read the spike data
channels = list()
for ii, this_file in enumerate(files):
    # load epochs data of one channel
    mat = loadmat(os.path.join(data_path, this_file),
                  struct_as_record=True, squeeze_me=False)

    # Store them per condition. Note that different channels may be recorded
    # from different epochs.
    channel = dict(('/'.join(cond), list())
                   for cond in product(classes, positions))
    data = mat['raster_data']
    ch_classes = [ii[0] for ii in mat['raster_labels'][0][0][0][0]]
    ch_positions = [ii[0] for ii in mat['raster_labels'][0][0][1][0]]
    for this_x, this_class, this_pos in zip(data, ch_classes, ch_positions):
        condition = '/'.join((this_class, this_pos))
        channel[condition].append(this_x)

    # Append data
    channels.append(channel)


# Reshape data such that we get a classic (n_epochs, n_channels, n_times)
# Note that this is a trick which can be misleading. Here, only four channels
# were recorded simulatenously.
n_rep = 19  # it should be 20, but there are some missing trials
X, y = list(), dict(cat=list(), position=list())

for this_class, this_pos in product(classes, positions):
    condition = '/'.join((this_class, this_pos))
    X.append(np.transpose([ch[condition][:n_rep]
                           for ch in channels], [1, 0, 2]))
    y['cat'].extend([this_class] * n_rep)
    y['position'].extend([this_pos] * n_rep)
X = np.vstack(X)
y = DataFrame(y)

###############################################################################
# Compute spike rate with a moving average by convolving a window of 20 time
# samples over the time axis
Xbin = np.apply_along_axis(lambda m: np.convolve(m, np.ones(20), mode='valid'),
                           axis=2, arr=X)

# Decimate time for faster computation
decim = 20
Xbin = Xbin[..., ::decim]

# Preprocess the categories
le = LabelEncoder()
y['condition'] = y[['cat', 'position']].apply(lambda x: '/'.join(x), axis=1)
y_condition = le.fit_transform(y['condition'])

##############################################################################
# Decode over time to identify when information peaks
clf = make_pipeline(StandardScaler(), LogisticRegression())
sl = SearchLight(clf, n_jobs=-1)

n_folds = 5
n_classes = len(set(y_condition))
n_epochs, n_channels, n_times = Xbin.shape
y_pred = np.zeros((n_epochs, n_times))
scores = np.zeros((n_folds, n_times))

cv = StratifiedKFold(n_folds)
scores = np.zeros((n_folds, n_times))
for fold, (train, test) in enumerate(cv.split(Xbin, y_condition)):
    sl.fit(Xbin[train], y_condition[train])
    # compute score for each time point
    scores[fold] = sl.score(Xbin[test], y_condition[test])

fig, ax = plt.subplots(1)
ax.plot(scores.mean(0))
ax.axhline(1. / n_classes, color='k', linestyle=':', label='chance')
ax.set_xlabel('Time samples')
ax.set_ylabel('Accuracy')
plt.show()

##############################################################################
# Representational Similarity Analysis is a neuroimaging-specific
# appelation to refer to statistics applied to the confusion matrix.
cv = StratifiedShuffleSplit(200)
toi = 32
cm = np.zeros((n_classes, n_classes))
for train, test in cv.split(Xbin, y_condition):
    clf.fit(Xbin[train, :, toi], y_condition[train])
    y_pred = clf.predict(Xbin[test, :, toi])
    cm += confusion_matrix(y_condition[test], y_pred)
cm /= sum(cm)

fig, ax = plt.subplots(1)
im = ax.matshow(cm, cmap='plasma')
ax.set_yticks(range(n_classes))
ax.set_yticklabels([''] + le.classes_)
ax.set_xticks(range(n_classes))
ax.set_xticklabels([''] + le.classes_, rotation=40, ha='left')

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
colors = dict(zip(classes, cmap(np.linspace(0., 1., len(classes)))))
labels = list()
for condition, (x, y) in zip(le.classes_, summary):
    class_, position = condition.split('/')
    label = None
    if class_ not in labels:
        label = class_
        labels.append(class_)
    plt.scatter(x, y, s=100, facecolors=colors[class_],
                label=label, edgecolors='k')
plt.axis('off')
plt.legend()
