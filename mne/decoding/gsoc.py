# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from .mixin import TransformerMixin


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
        # XXX Currently, we only accepts X as a 2D or 3D numpy array as but we
        # may eventually allow epochs too.

        # Recontruct epochs
        if (X.ndim == 3) and (self.n_chan is None):
            return X
        else:
            n_epoch = len(X)
            n_time = np.prod(X.shape[1:]) // self.n_chan
            X = np.reshape(X, [n_epoch, self.n_chan, n_time])
        return X

    def _check_init(self):
        if self.n_chan is not None or not isinstance(self.n_chan, int):
            raise ValueError('n_chan must be None or an integer, got %s'
                             'instead.' % self.n_chan)
