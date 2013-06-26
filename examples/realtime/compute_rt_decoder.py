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

from mne.realtime.classifier import ConcatenateChannels

scores_x, scores = [], []

pl.ion()

scaler = preprocessing.StandardScaler()
concatenator = ConcatenateChannels()
clf = SVC(C=1, kernel='linear')

scaled_classifier = Pipeline([('concat', concatenator),
                              ('scaler', scaler), ('svm', clf)])

for ev_num, ev in enumerate(rt_epochs.iter_evoked()):

    if ev_num == 0:
        X = ev.data[None, ...]
        y = int(ev.comment)
    else:
        X = np.concatenate((X, ev.data[None, ...]), axis=0)
        y = np.append(y, int(ev.comment))

    if ev_num >= min_trials:

        # Find number of trials in training and test set
        trnum = round(np.shape(X)[0]*tr_percent/100)
        tsnum = np.shape(X)[0] - trnum

        scaled_classifier = scaled_classifier.fit(X[:trnum], y[:trnum])
        scores.append(scaled_classifier.score(X[-tsnum:], y[-tsnum:])*100)

        scores_x.append(ev_num)

        # Plot accuracy
        pl.clf()

        pl.plot(scores_x[-5:], scores[-5:], '+',
                label="Classif. score")
        pl.hold(True)
        pl.plot(scores_x[-5:], scores[-5:])
        pl.axhline(50, color='k', linestyle='--', label="Chance level")
        pl.xlabel('Trials')
        pl.ylabel('Classification score (% correct)')
        pl.ylim([30, 105])
        pl.title('Real-time decoding')
        pl.show()

        # time.sleep() isn't used because of known issues with the Spyder
        pl.waitforbuttonpress(0.1)
