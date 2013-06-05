# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:58:27 2013

@author: mainakjas
"""

import sklearn
import numpy as np

class RtClassifier():

    def __init__(self, method, fit, predict, partial_fit):


        [Tr_X, Ts_X, Tr_Y, Ts_Y] = SplitData(self, epochs, Y, tr_percent)

        # Online training and testing
        clf = SVC(C=1, kernel='linear')
        clf.fit(Tr_X,Tr_Y)
        result = clf.predict(Ts_X)

    def SplitData(self, epochs, Y, tr_percent):

        trnum = round(np.shape(epochs)[0]*tr_percent/100)
        tsnum = np.shape(epochs)[0] - trnum

        Tr_X = np.reshape(epochs[:trnum,:,:],
                          [trnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Ts_X= np.reshape(epochs[-tsnum:,:,:],
                         [tsnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Tr_Y = Y[:trnum]
        Ts_Y = Y[-tsnum:]

        return Tr_X, Ts_X, Tr_Y, Ts_Y