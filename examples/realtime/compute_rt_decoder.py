"""
=======================
Decoding real-time data
=======================

Supervised machine learning applied to MEG data in sensor space.
Here the classifier is updated every 5 trials and the decoding
accuracy is plotted
"""
# Authors: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.realtime import RtClient, RtEpochs

import numpy as np
import pylab as pl

client = RtClient('localhost')
info = client.get_measurement_info()

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

total_trials = 50
tr_percent = 60  # Training %
min_trials = 10  # minimum trials after which decoding should start

# select gradiometers
picks = mne.fiff.pick_types(info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=info['bads'])

# create the real-time epochs object
rt_epochs = RtEpochs(client, event_id, tmin, tmax, total_trials,
                     consume_epochs=False, picks=picks, decim=1,
                     reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# Decoding in sensor space using a linear SVM
n_times = len(rt_epochs.times)

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, ShuffleSplit

from mne.realtime.classifier import ConcatenateChannels, FilterEstimator

scores_x, scores, std_scores = [], [], []

pl.ion()

filt = FilterEstimator(rt_epochs.info, 1, 40, verbose=False)
scaler = preprocessing.StandardScaler()
concatenator = ConcatenateChannels()
clf = SVC(C=1, kernel='linear')

concat_classifier = Pipeline([('filter', filt), ('concat', concatenator),
                              ('scaler', scaler), ('svm', clf)])

for ev_num, ev in enumerate(rt_epochs.iter_evoked()):

    if ev_num == 0:
        X = ev.data[None, ...]
        y = int(ev.comment)
    else:
        X = np.concatenate((X, ev.data[None, ...]), axis=0)
        y = np.append(y, int(ev.comment))

    if ev_num >= min_trials:

        cv = ShuffleSplit(len(y), 5, test_size=0.2, random_state=42)
        scores_t = cross_val_score(concat_classifier, X, y, cv=cv,
                                   n_jobs=1)*100

        std_scores.append(scores_t.std())
        scores.append(scores_t.mean())
        scores_x.append(ev_num)

        # Plot accuracy
        pl.clf()

        pl.plot(scores_x[-5:], scores[-5:], '+', label="Classif. score")
        pl.hold(True)
        pl.plot(scores_x[-5:], scores[-5:])
        pl.axhline(50, color='k', linestyle='--', label="Chance level")
        hyp_limits = (np.asarray(scores[-5:]) - np.asarray(std_scores[-5:]),
                      np.asarray(scores[-5:]) + np.asarray(std_scores[-5:]))
        pl.fill_between(scores_x[-5:], hyp_limits[0], y2=hyp_limits[1],
                        color='b', alpha=0.5)
        pl.xlabel('Trials')
        pl.ylabel('Classification score (% correct)')
        pl.ylim([30, 105])
        pl.title('Real-time decoding')
        pl.show()

        # time.sleep() isn't used because of known issues with the Spyder
        pl.waitforbuttonpress(0.1)
