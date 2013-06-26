# Authors: Romain Trachel <romain.trachel@inria.fr>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from .cov import compute_covariance, Covariance


class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)

    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    See [1].

    Parameters
    ----------
    n_components : int
        The maximum number of components.
    pick_components : None (default) or array of int
        Indices of components to decompose M/EEG signals
        (if None, all components are used).

    Attributes
    ----------
    filters_ : ndarray
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray
        If fit, the mean squared power for each component.
    std_ : ndarray
        If fit, the std squared power for each component.

    [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.
    """
    def __init__(self, n_components=64, pick_components=None):
        self.n_components = n_components
        self.pick_components = pick_components
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, epochs, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        epochs : ndarray
            The CSP is estimated on the concatenated epochs.
        y : array
            The classe for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")

        # concatenate epochs
        if len(epochs.shape) == 3:
            class_1 = np.transpose(epochs[y == classes[0]],
                                   [1, 0, 2]).reshape(epochs.shape[1], -1)
            class_2 = np.transpose(epochs[y == classes[1]],
                                   [1, 0, 2]).reshape(epochs.shape[1], -1)
        else:
            class_1 = epochs[:, y == classes[0]].copy()
            class_2 = epochs[:, y == classes[1]].copy()

        # fit on empirical covariance
        self._fit(np.dot(class_1, class_1.T),
                  np.dot(class_2, class_2.T))

        # compute sources for train and test data
        X = self.transform(epochs)

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def fit_epochs(self, epochs, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        epochs : list of Epochs
            The epochs for the 2 classes.
            The CSP is estimated on the Epochs objects.
        y : array
            The classe for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")

        covariance = [compute_covariance(epochs[y == c]).data for c in classes]
        # call decomposition algorithm
        self._fit(*covariance)
        
        # compute sources for train and test data
        X = self.transform_epochs(epochs, self.pick_components)
        
        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def fit_covariance(self, cov_list):
        """Run the CSP decomposition on covariance matrix.

        Parameters
        ----------
        cov_list : list of Covariance
            The Covariance of 2 classes.
            The CSP is estimated on the Covariance objects.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        if (not isinstance(cov_list, list) or len(cov_list) != 2 or
                not all([isinstance(c, Covariance) for c in cov_list])):
            raise ValueError('CSP decomposition needs a list of 2 '
                             'Covariance objects')

        covariance = [c.data.copy() for c in cov_list]

        # call decomposition algorithm
        self._fit(*covariance)
        return self

    def _fit(self, cov_a, cov_b):
        """ Aux Function (modifies cov_a and cov_b inplace)"""

        cov_a /= np.trace(cov_a)
        cov_b /= np.trace(cov_b)
        # computes the eigen values
        lambda_, u = linalg.eigh(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = lambda_[ind]

        u = u[:, ind]
        p = np.sqrt(linalg.pinv(np.diag(lambda2_))) * u.T

        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        g, b = linalg.eigh(w_a, w_b)
        # sort eigen values
        ind = np.argsort(g)
        b = b[:, ind]
        # and project
        w = np.dot(b.T, p)

        self.filters_ = w
        self.patterns_ = linalg.pinv(w).T

    def fit_transform(self, epochs, y):
        """Estimate the CSP decomposition on epochs and apply filters

        Parameters
        ----------
        epochs : ndarray
            The CSP is estimated on the concatenated epochs.
        y : array
            The classe for each epoch.

        Returns
        -------
        X : array
            Returns the data filtered by CSP
        """
        return self.fit(epochs, y).transform(epochs)

    def transform_epochs(self, epochs, pick_components=None):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        pick_components : array
            The list of CSP filters to estimate the sources.
            default is first and last components corresponding to the largest
            and the smallest eigen values.
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch.
        """
        if pick_components is None:
            pick_components = self.pick_components
        if pick_components is None:
            pick_components = slice(None, None, None)

        if not hasattr(self, 'filters_'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        if len(pick_components) > self.n_components:
            raise ValueError('picked components must be < n_components')

        pick_filters = self.filters_[pick_components]
        return np.asarray([np.dot(pick_filters, e.get_data()) for e in epochs])

    def transform(self, epochs, y=None):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : ndarray
            The sources are estimated for each epochs.
        
        Returns
        -------
        X : array
            Returns the data filtered by CSP
        """
        
        pick_filters = self.filters_[self.pick_components]
        # compute sources for train and test data
        X = np.asarray([np.dot(pick_filters, e) for e in epochs])
        
        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # Standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        X -= self.mean_
        X /= self.std_
        return X
