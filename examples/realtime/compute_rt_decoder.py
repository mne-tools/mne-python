# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:19:15 2013

@author: mainakjas
"""

print __doc__

import mne
from mne.realtime import RtClient, RtEpochs

import numpy as np

client = RtClient('localhost')
info = client.get_measurement_info()

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

minchunks = 20
trnum = minchunks*3
testnum = minchunks

# select gradiometers
picks = mne.fiff.pick_types(info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=info['bads'])

# create the real-time epochs object
rt_epochs = RtEpochs(client, event_id, tmin, tmax, minchunks,
                     consume_epochs=False, picks=picks, decim=1,
                     reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# Decoding in sensor space using a linear SVM
n_times = len(rt_epochs.times)

for ii in range(0,3):

    if (ii==0):
        epochs = rt_epochs._get_data_from_disk()
        events = np.asarray(rt_epochs.events)
    else:
        epochs = np.append(epochs,rt_epochs._get_data_from_disk(),axis=0)
        events = np.append(events,np.asarray(rt_epochs.events),axis=0)

    print events.shape
    rt_epochs.remove_old_epochs(minchunks)

X = np.reshape(epochs, [trnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
Y = events[:,2]

from sklearn.svm import SVC

clf = SVC(C=1, kernel='linear')
clf.fit(X,Y)

#rt_epochs.remove_old_epochs(m)

epochs_test = rt_epochs._get_data_from_disk()
X_test = np.reshape(epochs_test, [testnum, np.shape(epochs_test)[2]*np.shape(epochs_test)[1]])
Y_test = np.asarray(rt_epochs.events)[0:testnum,2]
result = clf.predict(X_test)

acc = sum(result==Y_test)/testnum*100