# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:19:15 2013

@author: mainakjas
"""

print __doc__

import mne
from mne.realtime import RtClient, RtEpochs

import numpy as np
import pylab as pl

client = RtClient('localhost')
info = client.get_measurement_info()

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

minchunks = 5
tr_percent = 60
min_trials = 20 # minimum trials after which decoding should start

#trnum = minchunks*10
#testnum = minchunks*4

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
from sklearn.svm import SVC

ii=0

while 1:

    if (ii==0):
        epochs = rt_epochs._get_data_from_disk()
        Y = np.asarray(rt_epochs.events)[0:minchunks,2]
    else:
        epochs = np.append(epochs,rt_epochs._get_data_from_disk(),axis=0)
        Y = np.append(Y,np.asarray(rt_epochs.events)[0:minchunks,2],axis=0)

    rt_epochs.remove_old_epochs(minchunks)

    if np.shape(epochs)[0] > min_trials:

        trnum = round(np.shape(epochs)[0]*tr_percent/100)
        tsnum = np.shape(epochs)[0] - trnum

        Tr_X = np.reshape(epochs[:trnum,:,:], [trnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Ts_X= np.reshape(epochs[-tsnum:,:,:], [tsnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Tr_Y = Y[:trnum]
        Ts_Y = Y[-tsnum:]

        clf = SVC(C=1, kernel='linear')
        clf.fit(Tr_X,Tr_Y)

        result = clf.predict(Ts_X)

        acc = sum(result==Ts_Y)/tsnum*100

        print "(train:test = %d:%d) :: (accuracy=%f)" % (trnum,tsnum,acc)

    ii += 1
#pl.plot(acc)