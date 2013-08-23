"""
==============================================
Real-time feedback for decoding :: Server Side
==============================================

This example demonstrates how to setup a real-time feedback
mechanism using StimServer and StimClient.

"""

print __doc__

# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import time
import mne

import numpy as np

from mne.datasets import sample
from mne.realtime import StimServer
from mne.realtime import MockRtClient

# Load fiff file to "fake data"
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.fiff.Raw(raw_fname, preload=True)

# Instantiating stimulation server
stim_server = StimServer(port=4218)
stim_server.start('localhost')

# Give time to start the client
time.sleep(10)

# The channels to be used while decoding
picks = mne.fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=raw.info['bads'])

rt_client = MockRtClient(raw)

# Importing modules for classification
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from mne.decoding import ConcatenateChannels, FilterEstimator

# Constructing the pipeline for classification
filt = FilterEstimator(raw.info, 1, 40)
scaler = preprocessing.StandardScaler()
concatenator = ConcatenateChannels()
clf = SVC(C=1, kernel='linear')

concat_classifier = Pipeline([('filter', filt), ('concat', concatenator),
                              ('scaler', scaler), ('svm', clf)])

# Just some initially decided events to be "faked"
# Rest will decided on the fly
ev_list = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]

scores, scores_x = [], []

for ii in range(50):

    # Tell the stim_client about the next stimuli
    stim_server.add_trigger(ev_list[ii])

    # Collecting data
    if ii == 0:
        X = rt_client.fake_data(event_id=ev_list[ii], tmin=-0.2,
                                tmax=0.5, picks=picks)[None, ...]
        y = ev_list[ii]
    else:
        X_temp = rt_client.fake_data(event_id=ev_list[ii], tmin=-0.2,
                                     tmax=0.5, picks=picks)[None, ...]

        X = np.concatenate((X, X_temp), axis=0)

        time.sleep(1)  # faking the isi
        y = np.append(y, ev_list[ii])

    # Start decoding after collecting sufficient data

    if ii >= 10:
        # Now start doing rtfeedback
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)

        y_pred = concat_classifier.fit(X_train, y_train).predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # do something if one class is decoded better than the other
        if cm[0, 0] > cm[1, 1]:
            print "We decoded class 1 better than class 3"
            ev_list.append(3)  # modifying future "faked data"
        else:
            print "We decoded class 3 better than class 1"
            ev_list.append(1)  # modifying future "faked data"

stim_server.shutdown()
