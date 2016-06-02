# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from .mixin import TransformerMixin
from ..epochs import _BaseEpochs


class EpochsTransformerMixin(TransformerMixin):
    def __init__(self, n_chan=None):
        self.n_chan = n_chan
        self._check_init()

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _reshape(self, X):
        """Recontruct epochs to get a n_trials * n_chan * n_time"""
        if isinstance(X, _BaseEpochs):
            X = X.get_data()
        elif not isinstance(X, np.ndarray):
            raise ValueError('X must be an Epochs or a 2D or 3D array, got '
                             '%s instead' % type(X))
        elif (X.ndim != 3) or (self.n_chan is None):
            n_epoch = len(X)
            n_time = np.prod(X.shape[1:]) // self.n_chan
            X = np.reshape(X, [n_epoch, self.n_chan, n_time])
        return X

    def _check_init(self):
        if self.n_chan is not None or not isinstance(self.n_chan, int):
            raise ValueError('n_chan must be None or an integer, got %s'
                             'instead.' % self.n_chan)
