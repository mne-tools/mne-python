# Authors: Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from . import Epochs
from .cov import compute_covariance, compute_raw_data_covariance
from .fiff.pick import pick_types

class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)
    algorithm 
    
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.

    Parameters
    ----------
    n_components : int
        The number of maximum components
    cov : instance of Covariance, sklearn.covariance or None
        The method to estimate covariance matrix of the signals
        if None MNE Covariance is used
    """
    
    def __init__(self, n_components=64, cov=None):
        
        self.n_components = n_components
        self.cov = cov
    
    
    def decompose_epochs(self, epochs, picks=None, verbose=None):
        """Run the CSP decomposition on epochs
        
        Parameters
        ----------
        epochs : list of Epochs
            The epochs for 2 classes. 
            The CSP is estimated on the concatenated epochs.
        picks : array-like
            Channels to be included relative to the channels already picked on
            epochs-initialization.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        
        if len(epochs) != 2:
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if not all([isinstance(e, Epochs) for e in epochs]):
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if picks is None:
            # just use epochs good data channels and avoid double picking
            picks = pick_types(epochs[0].info, include=epochs[0].ch_names,
                               exclude='bads')
        
        if self.cov is None:
            # compute covariance for class a
            cov_a = compute_covariance(epochs[0]).data
            cov_a /= np.trace(cov_a)
            # and for class b
            cov_b = compute_covariance(epochs[1]).data
            cov_b /= np.trace(cov_b)
        else:
            # compute concatenated epochs 
            data_a = np.transpose(epochs[0].get_data(),
                                  [1, 0, 2]).reshape(n, -1).T
            data_b = np.transpose(epochs[1].get_data(),
                                  [1, 0, 2]).reshape(n, -1).T
            # compute covariance for class a
            self.cov.fit(data_a)
            cov_a = self.cov.covariance_ / np.trace(cov.covariance_)
            # and for class b
            self.cov.fit(data_b)
            cov_b = self.cov.covariance_ / np.trace(cov.covariance_)
        
        # call decomposition algorithm
        self._decompose(cov_a,cov_b)
        return self
    
    
    def _decompose(self, cov_a, cov_b):
        """ Aux Function """
        # computes the eigen values
        (lambda_, u) = linalg.eig(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = np.sort(lambda_)[::-1]
        
        u  = u[:,ind]
        p  = np.sqrt(linalg.pinv(np.diag(lambda2_.real))) * u.T
        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        (g, b) = linalg.eig(w_a, w_b)
        # sort eigen values
        ind = np.argsort(g.real)
        b = b[:,ind].real
        # and project
        w = np.dot(b.T,p)
        
        self.filters  = w
        self.patterns = linalg.pinv(w).T
    
    
    def get_sources_epochs(self, epochs, pick_components = [0, -1]):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        pick_components : array
            The list of CSP filters to estimate the sources
            default is first and last components corresponding to the largest
            and the smallest eigen values
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
        if not hasattr(self, 'filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        
        if len(pick_components) > self.n_components:
            raise ValueError('picked components must be < n_components')
        
        return self._get_sources_epochs(epochs, pick_components)
    
    def _get_sources_epochs(self, epochs, pick_components):
        """ Aux Function """
        data = epochs.get_data()
        (nEpoch,nSensor,nTime) = data.shape
        csp_data = np.zeros_like(data)
        for i in np.arange(nEpoch):
            csp_data[i,:,:] = np.dot(self.filters,data[i,:,:])
        
        return csp_data[:,pick_components,:]
    
    
    def plot_sources_maps(self):
        """Plot CSP sources topographies
        """
        
