# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import numpy as np


class RtClassifier:

    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, estimator):

        self.estimator = estimator

    def split_data(self, epochs, Y, tr_percent):
        """Split data into training and test set

        Parameters
        ----------
        epochs :
        Y :
        tr_percent:

        Returns
        -------
        Tr_X:
        Ts_X:
        Tr_Y:
        Ts_Y:

        """

        trnum = round(np.shape(epochs)[0]*tr_percent/100)
        tsnum = np.shape(epochs)[0] - trnum

        Tr_X = np.reshape(epochs[:trnum, :, :],
                          [trnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Ts_X = np.reshape(epochs[-tsnum:, :, :],
                          [tsnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Tr_Y = Y[:trnum]
        Ts_Y = Y[-tsnum:]

        return Tr_X, Ts_X, Tr_Y, Ts_Y

    def fit(self, X, y):

        self.estimator.fit(X, y)
        return self

    def predict(self, X):

        result = self.estimator.predict(X)

        return result
