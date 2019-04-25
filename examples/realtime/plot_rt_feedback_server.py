"""
==============================================
Real-time feedback for decoding :: Server Side
==============================================

This example demonstrates how to setup a real-time feedback
mechanism using StimServer and StimClient.

The idea here is to display future stimuli for the class which
is predicted less accurately. This allows on-demand adaptation
of the stimuli depending on the needs of the classifier.

This will execute the ``rt_feedback_client.py`` script in a separate
process so that both can run concurrently.

All brain responses are simulated from a fiff file to make it easy
to test. However, it should be possible to adapt this script
for a real experiment.
"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import mne
from mne.datasets import sample
from mne.utils import running_subprocess
from mne.realtime import StimServer, MockRtClient
from mne.decoding import Vectorizer, FilterEstimator

print(__doc__)

# Load fiff file to simulate data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Instantiating stimulation server

# The with statement is necessary to ensure a clean exit
fig, ax = plt.subplots(1)
ax.set(xlabel='Trials', ylabel='Classification score (% correct)',
       title='Real-time feedback')
isi = 0.01  # this is unrealistic, but will make the example run quickly
n_trials = 40  # number of trials to simulate
n_start = 5  # number of trials to run before decoding
rng = np.random.RandomState(0)
with StimServer(port=4218) as stim_server:

    # The channels to be used while decoding
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude=raw.info['bads'])

    rt_client = MockRtClient(raw)

    # Constructing the pipeline for classification
    # don't highpass filter because of short signal length of epochs
    filt = FilterEstimator(raw.info, None, 40,
                           # keep all channels that are picked from the
                           # RtClient
                           picks=np.arange(len(picks), dtype=int))
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    clf = SVC(C=1, kernel='linear')
    concat_classifier = Pipeline([('filter', filt), ('vector', vectorizer),
                                  ('scaler', scaler), ('svm', clf)])
    ev_list = list(rng.randint(3, 5, n_start))  # some random starting events
    score_c1, score_c2, score_x = [], [], []

    command = [sys.executable, 'rt_feedback_client.py']
    with running_subprocess(command, after='terminate',
                            stdout=None, stderr=None):
        stim_server.start(verbose=True)
        for ii in range(n_trials):
            # Tell the stim_client about the next stimuli
            stim_server.add_trigger(ev_list[ii])

            # Collecting data
            if ii == 0:
                X = rt_client.get_event_data(event_id=ev_list[ii], tmin=-0.2,
                                             tmax=0.5, picks=picks,
                                             stim_channel='STI 014')[None, ...]
                y = ev_list[ii]
            else:
                X_temp = rt_client.get_event_data(
                    event_id=ev_list[ii], tmin=-0.2, tmax=0.5, picks=picks,
                    stim_channel='STI 014')
                X_temp = X_temp[np.newaxis]
                X = np.concatenate((X, X_temp), axis=0)
                time.sleep(isi)  # simulating the isi
                y = np.append(y, ev_list[ii])

            # Start decoding after collecting sufficient data
            if ii >= n_start - 1:
                # Now start doing rtfeedback
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=rng)
                y_pred = concat_classifier.fit(X_train,
                                               y_train).predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                score_c1.append(float(cm[0, 0]) / sum(cm, 1)[0] * 100)
                score_c2.append(float(cm[1, 1]) / sum(cm, 1)[1] * 100)
                score_x.append(ii + 1)

                # do something if one class is decoded better than the other
                if score_c1[-1] < score_c2[-1]:
                    ev_list.append(3)  # adding more LV to future simulations
                else:
                    ev_list.append(4)  # adding more RV to future simulations
                print('Trial %d accuracy: %0.1f%%'
                      % (ii + 1, np.mean([score_c1[-1], score_c2[-1]])))

                # Now plot the accuracy
                lh = ax.plot(score_x[-2:], score_c1[-2:],
                             c='r', marker='o', ls='-')[0]
                rh = ax.plot(score_x[-2:], score_c2[-2:],
                             c='b', marker='o', ls='-')[0]
                ax.set(ylim=[0, 100], xticks=score_x,
                       xlim=[score_x[0], score_x[-1]])
                ax.legend((lh, rh), ('LV', 'RV'), loc='upper left')
                plt.draw()
                plt.pause(0.01)
