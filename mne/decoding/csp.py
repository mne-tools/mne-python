# Authors: Romain Trachel <romain.trachel@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from distutils.version import LooseVersion

from .mixin import TransformerMixin


class CSP(TransformerMixin):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)

    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    See [1].

    Parameters
    ----------
    n_components : int, default 4
        The number of components to decompose M/EEG signals.
        This number should be set by cross-validation.
    reg : float, str, None
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('lws') or
                Oracle Approximating Shrinkage ('oas')
    log : bool
        If true, apply log to standardize the features.
        If false, features are just z-scored.

    Attributes
    ----------
    `filters_` : ndarray
        If fit, the CSP components used to decompose the data, else None.
    `patterns_` : ndarray
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    `mean_` : ndarray
        If fit, the mean squared power for each component.
    `std_` : ndarray
        If fit, the std squared power for each component.

    [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.
    """
    def __init__(self, n_components=4, reg=None, log=True):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, epochs_data, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data to estimate the CSP on.
        y : array
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")
        # concatenate epochs
        class_1 = np.transpose(epochs_data[y == classes[0]],
                               [1, 0, 2]).reshape(epochs_data.shape[1], -1)
        class_2 = np.transpose(epochs_data[y == classes[1]],
                               [1, 0, 2]).reshape(epochs_data.shape[1], -1)
        if self.reg is None:
            # compute empirical covariance
            cov_1 = np.dot(class_1, class_1.T)
            cov_2 = np.dot(class_2, class_2.T)
        else:
            # use sklearn covariance estimators
            if isinstance(self.reg, float):
                if (self.reg < 0) or (self.reg > 1):
                    raise ValueError('0 <= shrinkage <= 1 for '
                                     'covariance regularization.')
                try:
                    import sklearn
                    sklearn_version = LooseVersion(sklearn.__version__)
                    from sklearn.covariance import ShrunkCovariance
                except ImportError:
                    raise Exception('the scikit-learn package is missing and '
                                    'required for covariance regularization.')
                if sklearn_version < '0.12':
                    skl_cov = ShrunkCovariance(shrinkage=self.reg,
                                               store_precision=False)
                else:
                    # init sklearn.covariance.ShrunkCovariance estimator
                    skl_cov = ShrunkCovariance(shrinkage=self.reg,
                                               store_precision=False,
                                               assume_centered=True)
            elif isinstance(self.reg, str):
                if self.reg == 'lws':
                    try:
                        from sklearn.covariance import LedoitWolf
                    except ImportError:
                        raise Exception('the scikit-learn package is missing '
                                        'and required for regularization.')
                    # init sklearn.covariance.LedoitWolf estimator
                    skl_cov = LedoitWolf(store_precision=False,
                                         assume_centered=True)
                elif self.reg == 'oas':
                    try:
                        from sklearn.covariance import OAS
                    except ImportError:
                        raise Exception('the scikit-learn package is missing '
                                        'and required for regularization.')
                    # init sklearn.covariance.OAS estimator
                    skl_cov = OAS(store_precision=False,
                                  assume_centered=True)
                else:
                    raise ValueError("regularization parameter should be "
                                     "of type str (got %s)." % type(self.reg))
            else:
                raise ValueError("regularization parameter should be "
                                 "of type str (got %s)." % type(self.reg))

            # compute regularized covariance using sklearn
            cov_1 = skl_cov.fit(class_1.T).covariance_
            cov_2 = skl_cov.fit(class_2.T).covariance_

        # then fit on covariance
        self._fit(cov_1, cov_2)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, e) for e in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def _fit(self, cov_a, cov_b):
        """Aux Function (modifies cov_a and cov_b in-place)"""

        cov_a /= np.trace(cov_a)
        cov_b /= np.trace(cov_b)
        # computes the eigen values
        lambda_, u = linalg.eigh(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = lambda_[ind]

        u = u[:, ind]
        p = np.dot(np.sqrt(linalg.pinv(np.diag(lambda2_))), u.T)

        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        vals, vecs = linalg.eigh(w_a, w_b)
        # sort vectors by discriminative power using eigen values
        ind = np.argsort(np.maximum(vals, 1. / vals))[::-1]
        vecs = vecs[:, ind]
        # and project
        w = np.dot(vecs.T, p)

        self.filters_ = w
        self.patterns_ = linalg.pinv(w).T

    def transform(self, epochs_data, y=None):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs_data : array, shape=(n_epochs, n_channels, n_times)
            The data.
        Returns
        -------
        X : ndarray of shape (n_epochs, n_sources)
            The CSP features averaged over time.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, e) for e in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)
        if self.log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X
