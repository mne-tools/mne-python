# Authors: Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import sklearn
from sklearn import covariance
import numpy as np
from numpy import linalg
import scipy.linalg

class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)
    algorithm 
    
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in binary decoding problems.

    Parameters
    ----------
    """
    
    @verbose
    def __init__(self, components, n_components=64, cov_func=None, 
                 random_state=None, verbose=None):
        
        if components is not None and \
                len(components) > n_csp_components:
            raise ValueError('components number must be smaller than '
                             'n_components')
        
        self.components   = components
        self.n_components = n_components
        self.cov_func = cov_func

    
    @verbose
    def decompose_epochs(self, epochs_a, epochs_b, cov_func = None,
                         picks=None, verbose=None):
        """Run the CSP decomposition on epochs
        
        Parameters
        ----------
        epochs_a : instance of Epochs
            The epochs for class a. 
            The CSP is estimated on the concatenated epochs.
        epochs_b : instance of Epochs for class b
            The epochs for class b. 
            The CSP is estimated on the concatenated epochs.
        cov_func : instance of sklearn.covariance
            The CSP can estimate covariance matrices using the sklearn package
            or empirically (if cov_func = None)
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
        self._decompose(epochs_a.get_data(),epochs_b.get_data(),self.cov_func)
        return self
    
    @verbose
    def _decompose(self, data_a, data_b, 
               cov = covariance.EmpiricalCovariance(assume_centered = True)):
        
        n = data_a.shape[1]
        
        # Compute concatenated epochs covariance matrices
        cov.fit(np.transpose(data_a,[1, 0, 2]).reshape(n, -1).T)
        cov_a = np.matrix(cov.covariance_ / np.trace(cov.covariance_))
        
        cov.fit(np.transpose(data_b,[1, 0, 2]).reshape(n, -1).T)
        cov_b = np.matrix(cov.covariance_ / np.trace(cov.covariance_))
        
        # computes the eigen values
        (Lambda,U)  = linalg.eig(cov_a + cov_b)
        # sort them
        ind    = np.argsort(Lambda)[::-1]
        Lambda = np.sort(Lambda)[::-1]
        
        U  = U[:,ind]
        P  = np.sqrt(linalg.pinv(np.diag(Lambda.real)))*U.T
        # Compute the generalized eigen value problem
        Wa = P*cov_a*P.T
        Wb = P*cov_b*P.T
        # and solve it
        (G,B) = scipy.linalg.eig(Wa,Wb)
        # sort eigen values
        ind = np.argsort(G.real)
        B   = B[:,ind].real
        # and project
        W = B.T*P
        
        self.csp_filters  = W
        self.csp_patterns = linalg.pinv(W).T
    
    
    def get_sources_epochs(self, epochs, components):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        components : array of shape (n_components)
            The list of CSP filters to estimate the sources
        
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
        if not hasattr(self, 'csp_filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        return self._get_sources_epochs(epochs, components)
    
    def _get_sources_epochs(self, epochs, components):
        
        data = epochs.get_data()
        (nEpoch,nSensor,nTime) = data.shape
        csp_data = np.zeros([nEpoch,components.shape[0],nTime])
        for i in np.arange(nEpoch):
            csp_data[i,:,:] = np.dot(self.csp_filters[components,:],data[i,:,:])
        
        return csp_data
        
        
