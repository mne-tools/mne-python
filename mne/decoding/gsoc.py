# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Asish Panda <asishrocks95@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from .mixin import TransformerMixin
from ..epochs import _BaseEpochs


class _EpochsTransformerMixin(TransformerMixin):
    """Mixin class for reshaping data to Epoch's standard shape

    This class is meant to be inherited by transformers that are to be
    used in scikit-learn pipeline. It provides functionality to convert
    data matrix into 3D.

    Parameters
    ----------
    n_chan : int (default : None)
        The number of channels. Used for reshaping data matrix into 3D.
        If none the matrix should be 3D else error is raised.
    """

    def __init__(self, n_chan=None):
        self.n_chan = n_chan
        self._check_init()

    def fit(self, X, y=None):
        """No use here. Added for scikit-learn compatibility.

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be reshaped into 3D. `n_chan` is used in 3D or 4D
            matrix.
        y : None
            Used for scikit-learn compatibility

        Returns
        -------
        self : Instance of EpochsTransformerMixin
            Return the same object.
        """
        return self

    def transform(self, X):
        """No use here. Added for scikit-learn compatibility.

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be reshaped into 3D. `n_chan` is used in 3D or 4D
            matrix.

        Returns
        -------
        X : numpy ndarray
            The same array.
        """
        return X

    def _reshape(self, X):
        """Recontruct epochs to get a n_trials * n_chan * n_time

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be reshaped into 3D. `n_chan` is used in 3D or 4D
            matrix.

        Returns
        -------
        X : numpy ndarray of shape (n_trials, n_chan, n_times)
            Transformed data.
        """
        if isinstance(X, _BaseEpochs):
            X = X.get_data()
            # TODO: pick data channels (EEG/MEG/SEEG/ECOG if epochs)
        elif not isinstance(X, np.ndarray):
            raise ValueError('X must be an Epochs or a 2D or 3D array, got '
                             '%s instead' % type(X))
        elif (X.ndim != 3) and (self.n_chan is None):
            raise ValueError("n_chan must be provided to convert it to 3D")
        elif (X.ndim != 3) or (self.n_chan is not None):
            n_epoch = len(X)
            n_time = np.prod(X.shape[1:]) // self.n_chan
            X = np.reshape(X, [n_epoch, self.n_chan, n_time])
        return X

    def _check_init(self):
        if self.n_chan is not None and not isinstance(self.n_chan, int):
            raise ValueError('n_chan must be None or an integer, got %s '
                             'instead.' % self.n_chan)


class UnsupervisedSpatialFilter(_EpochsTransformerMixin):
    """Fit and transform with an unsupervised spatial filtering across time
    and samples.

    Parameters
    ----------
    estimator : scikit-learn estimator
        Estimator using some decomposition algorithm.
    n_chan : int | None
        The number of channels.
    """
    def __init__(self, estimator, n_chan=None):
        self.n_chan = n_chan
        self._check_init()
        self.estimator = estimator
        for attr in ['fit', 'transform', 'fit_transform']:
            if not hasattr(estimator, attr):
                raise ValueError('estimator must be a sklearn transformer')

    def fit(self, X, y=None):
        """Make the data compatibile with scikit-learn estimator

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be filtered.
        y : None
            Used for scikit-learn compatibility.

        Returns
        -------
        self : Instance of UnsupervisedSpatialFilter
            Return the modified instance.
        """
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        self.estimator.fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Transform the data to its filtered components after fitting

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be reshaped.
        y : None
            Used for scikit-learn compatibility.

        Returns
        -------
        X : numpy ndarray of shape(n_trials, n_chan, n_times)
            The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """Transform the data to its spatial filters.

        Parameters
        ----------
        X : numpy array of dimensions [2,3,4]
            The data to be reshaped.

        Returns
        -------
        X : numpy ndarray of shape(n_trials, n_chan, n_times)
            The transformed data.
        """
        X = self._reshape(X)
        n_epoch, n_chan, n_time = X.shape
        # trial as time samples
        X = np.transpose(X, [1, 0, 2]).reshape([n_chan, n_epoch * n_time]).T
        X = self.estimator.transform(X)
        X = np.reshape(X.T, [-1, n_epoch, n_time]).transpose([1, 0, 2])
        return X
