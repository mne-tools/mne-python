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


class UnsupervisedSpatialFilter(EpochsTransformerMixin):
    """Fit and transform with an unsupervised spatial filtering across time
    and samples.

    e.g.
    filter = UnsupervisedSpatialFilter(PCA())
    filter.fit_transform(X, y=None)
    """
    def __init__(self, estimator, n_chan=None):
        self.n_chan = n_chan
        if self.n_chan is not None and not isinstance(self.n_chan, int):
            raise ValueError('n_chan must be None or an int, got %s '
                             'instead' % type(n_chan))
        self.estimator = estimator
        if not isinstance(estimator, TransformerMixin):
            # XXX just check if has fit transform etc attributes?
            raise ValueError('estimator must be a scikit-learn transformer')

    def fit(self, X, y=None):
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        self.estimator.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        X = self.estimator.transform(X)
        X = np.reshape(X.T, [-1, n_epoch, n_time]).transpose([1, 0, 2])
        return X
